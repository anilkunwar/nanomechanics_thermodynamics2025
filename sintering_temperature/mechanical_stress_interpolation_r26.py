# =============================================
# PUBLICATION-QUALITY CHART VISUALIZER
# =============================================

class PublicationChartVisualizer:
    """Specialized visualizer for publication-quality figures with larger labels"""
    
    def __init__(self):
        # Publication quality settings
        self.publication_params = {
            'font_family': 'Arial',
            'title_font_size': 24,
            'axis_label_font_size': 20,
            'tick_label_font_size': 16,
            'legend_font_size': 16,
            'line_width': 3.0,
            'marker_size': 10,
            'dpi': 300,
            'figure_size': (14, 10)
        }
        
        # Color schemes optimized for publications
        self.pub_colors = {
            'stress_colors': {
                'sigma_hydro': '#1f77b4',  # Blue
                'von_mises': '#ff7f0e',    # Orange
                'sigma_mag': '#2ca02c',    # Green
            },
            'defect_colors': {
                'ISF': '#e377c2',          # Pink
                'ESF': '#7f7f7f',          # Gray
                'Twin': '#9467bd',         # Purple
                'No Defect': '#17becf',    # Cyan
            },
            'system_colors': {
                'System 1 (Perfect Crystal)': '#2ca02c',
                'System 2 (Stacking Faults/Twins)': '#ff7f0e',
                'System 3 (Plastic Deformation)': '#d62728'
            }
        }
    
    def set_publication_params(self, fig, ax=None, title=None, xlabel=None, ylabel=None):
        """Apply publication-quality parameters to matplotlib figure"""
        params = self.publication_params
        
        if title:
            fig.suptitle(title, fontsize=params['title_font_size'], 
                        fontweight='bold', fontfamily=params['font_family'])
        
        if ax is not None:
            if xlabel:
                ax.set_xlabel(xlabel, fontsize=params['axis_label_font_size'], 
                            fontfamily=params['font_family'], fontweight='bold')
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=params['axis_label_font_size'], 
                            fontfamily=params['font_family'], fontweight='bold')
            
            # Set tick parameters
            ax.tick_params(axis='both', which='major', 
                          labelsize=params['tick_label_font_size'])
            ax.tick_params(axis='both', which='minor', 
                          labelsize=params['tick_label_font_size'] - 2)
            
            # Set legend if present
            if ax.get_legend():
                ax.legend(fontsize=params['legend_font_size'], 
                         loc='best', framealpha=0.9)
        
        # Tight layout for better spacing
        fig.tight_layout()
        
        return fig
    
    def create_publication_sintering_plot(self, stresses, temperatures, defect_type='Twin',
                                         title="Publication: Sintering Temperature vs Hydrostatic Stress"):
        """Create publication-quality sintering temperature plot"""
        
        params = self.publication_params
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=params['figure_size'], dpi=params['dpi'])
        
        # 1. Main sintering curve
        ax1.plot(stresses, temperatures, color=self.pub_colors['stress_colors']['sigma_hydro'], 
                linewidth=params['line_width'], label='Exponential Model', marker='o', 
                markersize=params['marker_size'])
        
        # System boundaries with enhanced visibility
        ax1.axvspan(0, 5, alpha=0.15, color=self.pub_colors['system_colors']['System 1 (Perfect Crystal)'], 
                   label='System 1 (Perfect)')
        ax1.axvspan(5, 20, alpha=0.15, color=self.pub_colors['system_colors']['System 2 (Stacking Faults/Twins)'], 
                   label='System 2 (SF/Twin)')
        ax1.axvspan(20, 35, alpha=0.15, color=self.pub_colors['system_colors']['System 3 (Plastic Deformation)'], 
                   label='System 3 (Plastic)')
        
        # Apply publication parameters
        self.set_publication_params(
            fig, ax1,
            xlabel='Absolute Hydrostatic Stress |σ_h| (GPa)',
            ylabel='Sintering Temperature (K)',
            title=title
        )
        ax1.grid(True, alpha=0.3, linewidth=1.5)
        ax1.legend(fontsize=params['legend_font_size'], loc='upper right')
        
        # 2. Temperature in Celsius for better readability
        temps_celsius = np.array(temperatures) - 273.15
        ax2.plot(stresses, temps_celsius, color=self.pub_colors['stress_colors']['sigma_hydro'], 
                linewidth=params['line_width'], marker='s', markersize=params['marker_size'])
        
        # Mark important temperature points
        important_temps = [200, 250, 300, 350]  # °C
        for temp_c in important_temps:
            idx = np.argmin(np.abs(temps_celsius - temp_c))
            if idx < len(stresses):
                ax2.annotate(f'{temp_c}°C', xy=(stresses[idx], temps_celsius[idx]),
                           xytext=(stresses[idx]+2, temps_celsius[idx]+10),
                           fontsize=params['legend_font_size'] - 2,
                           arrowprops=dict(arrowstyle='->', color='gray', linewidth=2))
        
        self.set_publication_params(
            fig, ax2,
            xlabel='Absolute Hydrostatic Stress |σ_h| (GPa)',
            ylabel='Sintering Temperature (°C)',
            title=f'Sintering Temperature in Celsius - {defect_type}'
        )
        ax2.grid(True, alpha=0.3, linewidth=1.5)
        
        # 3. Temperature reduction percentage
        T0 = 623.0  # Reference temperature
        temp_reduction = ((T0 - np.array(temperatures)) / T0) * 100
        
        ax3.plot(stresses, temp_reduction, color='#d62728', 
                linewidth=params['line_width'], marker='^', markersize=params['marker_size'])
        
        # Highlight reduction milestones
        for reduction in [20, 40, 60]:
            idx = np.argmin(np.abs(temp_reduction - reduction))
            if idx < len(stresses) and temp_reduction[idx] >= reduction:
                ax3.annotate(f'{reduction:.0f}% reduction', 
                           xy=(stresses[idx], temp_reduction[idx]),
                           xytext=(stresses[idx]+1, temp_reduction[idx]+5),
                           fontsize=params['legend_font_size'] - 2,
                           arrowprops=dict(arrowstyle='->', color='gray', linewidth=2))
        
        self.set_publication_params(
            fig, ax3,
            xlabel='Absolute Hydrostatic Stress |σ_h| (GPa)',
            ylabel='Temperature Reduction (%)',
            title=f'Temperature Reduction - {defect_type}'
        )
        ax3.grid(True, alpha=0.3, linewidth=1.5)
        
        # 4. Stress-temperature correlation
        ax4.scatter(stresses, temperatures, c=stresses, cmap='viridis', 
                   s=params['marker_size']*20, alpha=0.7, edgecolors='black', linewidth=2)
        
        # Add trend line
        z = np.polyfit(stresses, temperatures, 2)
        p = np.poly1d(z)
        stress_fit = np.linspace(min(stresses), max(stresses), 100)
        temp_fit = p(stress_fit)
        ax4.plot(stress_fit, temp_fit, 'r--', linewidth=params['line_width']*1.5, 
                label='Quadratic Fit')
        
        self.set_publication_params(
            fig, ax4,
            xlabel='Absolute Hydrostatic Stress |σ_h| (GPa)',
            ylabel='Sintering Temperature (K)',
            title=f'Stress-Temperature Correlation - {defect_type}'
        )
        ax4.grid(True, alpha=0.3, linewidth=1.5)
        ax4.legend(fontsize=params['legend_font_size'])
        
        # Add colorbar for scatter plot
        scatter = ax4.collections[0]
        cbar = fig.colorbar(scatter, ax=ax4)
        cbar.set_label('Hydrostatic Stress (GPa)', fontsize=params['axis_label_font_size']-2)
        cbar.ax.tick_params(labelsize=params['tick_label_font_size']-2)
        
        plt.suptitle(f'{title} - Defect Type: {defect_type}', 
                    fontsize=params['title_font_size'] + 2, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        return fig
    
    def create_publication_radar_chart(self, angles, stress_components, 
                                      title="Publication: Stress Component Radar",
                                      labels=None):
        """Create publication-quality radar chart with enhanced visibility"""
        
        params = self.publication_params
        
        if labels is None:
            labels = list(stress_components.keys())
        
        # Number of variables
        N = len(angles)
        
        # What will be the angle of each axis in the plot
        angles_rad = np.linspace(0, 2 * np.pi, N, endpoint=False)
        
        # Close the loop
        angles_rad_closed = np.concatenate((angles_rad, [angles_rad[0]]))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10), dpi=params['dpi'], 
                              subplot_kw=dict(projection='polar'))
        
        # Plot each stress component
        for idx, (comp_name, values) in enumerate(stress_components.items()):
            if len(values) == N:
                values_closed = np.concatenate((values, [values[0]]))
                
                # Choose color from palette
                if comp_name in self.pub_colors['stress_colors']:
                    color = self.pub_colors['stress_colors'][comp_name]
                else:
                    color = plt.cm.tab20(idx)
                
                ax.plot(angles_rad_closed, values_closed, 'o-', linewidth=params['line_width'],
                       markersize=params['marker_size'], label=comp_name.replace('_', ' ').title(),
                       color=color)
                
                # Fill area with transparency
                ax.fill(angles_rad_closed, values_closed, alpha=0.1, color=color)
        
        # Set labels for each axis (convert radians to degrees for display)
        angle_labels = [f'{ang:.0f}°' for ang in angles]
        angle_labels.append(angle_labels[0])  # Close the loop
        
        ax.set_xticks(angles_rad_closed)
        ax.set_xticklabels(angle_labels, fontsize=params['tick_label_font_size'], 
                          fontweight='bold')
        
        # Set radial labels
        ax.set_rlabel_position(0)
        ax.tick_params(axis='y', labelsize=params['tick_label_font_size'])
        
        # Add title and legend
        ax.set_title(title, fontsize=params['title_font_size'], fontweight='bold', 
                    pad=20, fontfamily=params['font_family'])
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), 
                 fontsize=params['legend_font_size'], framealpha=0.9)
        
        # Add grid with enhanced visibility
        ax.grid(True, linewidth=1.5, alpha=0.7)
        
        # Add radial lines for better readability
        ax.set_rgrids([], angle=0)  # Clear default grids
        for r in np.linspace(0, ax.get_rmax(), 6):
            ax.plot(angles_rad_closed, [r] * len(angles_rad_closed), 
                   '--', color='gray', alpha=0.3, linewidth=1)
        
        # Add habit plane marker if within range
        habit_angle_deg = 54.7
        if min(angles) <= habit_angle_deg <= max(angles):
            habit_angle_rad = np.radians(habit_angle_deg)
            ax.axvline(x=habit_angle_rad, color='green', linestyle='--', 
                      linewidth=3, alpha=0.7, label='Habit Plane (54.7°)')
            
            # Add annotation
            ax.annotate('Habit Plane', xy=(habit_angle_rad, ax.get_rmax()*0.8),
                       xytext=(habit_angle_rad + 0.3, ax.get_rmax()*0.85),
                       fontsize=params['legend_font_size'], fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='green', linewidth=2))
        
        plt.tight_layout()
        return fig

# =============================================
# ENHANCED HABIT PLANE VISUALIZER WITH PUBLICATION SETTINGS
# =============================================

class EnhancedHabitPlaneVisualizer(HabitPlaneVisualizer):
    """Enhanced visualizer with publication-quality settings"""
    
    def __init__(self, habit_angle=54.7):
        super().__init__(habit_angle)
        self.pub_visualizer = PublicationChartVisualizer()
        
        # Enhanced color schemes for better visibility
        self.enhanced_stress_colors = {
            'sigma_hydro': 'rgb(31, 119, 180)',
            'von_mises': 'rgb(255, 127, 14)',
            'sigma_mag': 'rgb(44, 160, 44)',
            'sigma_xx': 'rgb(214, 39, 40)',
            'sigma_yy': 'rgb(148, 103, 189)',
            'sigma_zz': 'rgb(140, 86, 75)'
        }
        
        self.enhanced_defect_colors = {
            'ISF': 'rgb(227, 119, 194)',   # Brighter pink
            'ESF': 'rgb(127, 127, 127)',   # Medium gray
            'Twin': 'rgb(188, 189, 34)',   # Olive green
            'No Defect': 'rgb(23, 190, 207)'  # Cyan
        }
    
    def create_enhanced_vicinity_sunburst(self, angles, stresses, stress_component='sigma_hydro',
                                         title="Enhanced Habit Plane Vicinity Analysis", 
                                         publication_quality=False):
        """Create sunburst chart with enhanced visibility options"""
        
        # Check if data is empty
        if len(angles) == 0 or len(stresses) == 0:
            return self.create_vicinity_sunburst(angles, stresses, stress_component, title)
        
        # Use base method if not publication quality
        if not publication_quality:
            return self.create_vicinity_sunburst(angles, stresses, stress_component, title)
        
        # Enhanced publication quality version
        angles = np.array(angles)
        stresses = np.array(stresses)
        
        # Filter to vicinity (habit_angle ± 45°)
        vicinity_range = 45.0
        mask = (angles >= self.habit_angle - vicinity_range) & (angles <= self.habit_angle + vicinity_range)
        vic_angles = angles[mask]
        vic_stresses = stresses[mask]
        
        if len(vic_angles) == 0:
            return self.create_vicinity_sunburst(angles, stresses, stress_component, title)
        
        # Create enhanced polar plot with larger labels
        fig = go.Figure()
        
        # Add main stress distribution with enhanced styling
        fig.add_trace(go.Scatterpolar(
            r=vic_stresses,
            theta=vic_angles,
            mode='markers+lines',
            marker=dict(
                size=12,  # Larger markers
                color=vic_stresses,
                colorscale='RdBu_r',  # Reversed for better contrast
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text=f"{stress_component.replace('_', ' ').title()} (GPa)",
                        font=dict(size=18, family="Arial", color='black')
                    ),
                    x=1.15,
                    thickness=25,
                    len=0.8,
                    tickfont=dict(size=14)
                ),
                line=dict(width=2, color='black'),
                symbol='circle'  # Clear symbol
            ),
            line=dict(color='rgba(50, 50, 50, 0.5)', width=2.5),
            name='Stress Distribution',
            hovertemplate='<b>Orientation</b>: %{theta:.2f}°<br><b>Stress</b>: %{r:.4f} GPa<extra></extra>'
        ))
        
        # Highlight habit plane with enhanced marker
        habit_idx = np.argmin(np.abs(vic_angles - self.habit_angle))
        if habit_idx < len(vic_stresses):
            habit_stress = vic_stresses[habit_idx]
            fig.add_trace(go.Scatterpolar(
                r=[habit_stress],
                theta=[vic_angles[habit_idx]],
                mode='markers+text',
                marker=dict(
                    size=30,  # Larger marker
                    color='rgb(46, 204, 113)',
                    symbol='star',
                    line=dict(width=3, color='black')
                ),
                text=['HABIT PLANE'],
                textposition='top center',
                textfont=dict(size=18, color='black', family='Arial Black', weight='bold'),
                name=f'Habit Plane ({self.habit_angle}°)',
                hovertemplate=f'<b>Habit Plane</b> ({self.habit_angle}°)<br><b>Stress</b>: {habit_stress:.4f} GPa<extra></extra>'
            ))
        
        # Enhanced layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=24, family="Arial Black", color='darkblue'),
                x=0.5,
                y=0.95
            ),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    gridcolor="rgba(100, 100, 100, 0.4)",
                    gridwidth=2,
                    linecolor="black",
                    linewidth=3,
                    tickfont=dict(size=16, color='black', family='Arial'),
                    title=dict(
                        text=f"{stress_component.replace('_', ' ').title()} (GPa)",
                        font=dict(size=20, color='black', family='Arial', weight='bold')
                    ),
                    range=[0, max(vic_stresses) * 1.3]
                ),
                angularaxis=dict(
                    gridcolor="rgba(100, 100, 100, 0.4)",
                    gridwidth=2,
                    linecolor="black",
                    linewidth=3,
                    rotation=90,
                    direction="clockwise",
                    tickmode='array',
                    tickvals=list(range(int(vic_angles[0]), int(vic_angles[-1]) + 1, 15)),
                    ticktext=[f'{i}°' for i in range(int(vic_angles[0]), int(vic_angles[-1]) + 1, 15)],
                    tickfont=dict(size=16, color='black', family='Arial'),
                    period=360,
                    thetaunit="degrees"
                ),
                bgcolor="rgba(240, 240, 240, 0.5)",
                sector=[vic_angles[0], vic_angles[-1]]
            ),
            showlegend=True,
            legend=dict(
                x=1.25,
                y=0.5,
                bgcolor='rgba(255, 255, 255, 0.95)',
                bordercolor='black',
                borderwidth=2,
                font=dict(size=16, family='Arial'),
                itemwidth=40
            ),
            width=1000,
            height=800,
            margin=dict(l=150, r=250, t=150, b=150),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_enhanced_defect_comparison_plot(self, defect_comparison, stress_component='sigma_hydro',
                                              title="Enhanced Defect Type Comparison",
                                              publication_quality=False):
        """Create enhanced comparison plot for different defect types"""
        
        # Use base method if not publication quality
        if not publication_quality:
            return self.create_defect_comparison_plot(defect_comparison, stress_component, title)
        
        # Enhanced publication quality version
        fig = go.Figure()
        
        if not defect_comparison:
            return self.create_defect_comparison_plot(defect_comparison, stress_component, title)
        
        # Add traces for each defect type with enhanced styling
        for defect_key, data in defect_comparison.items():
            if not isinstance(data, dict):
                continue
                
            if 'angles' in data and 'stresses' in data and stress_component in data['stresses']:
                defect_type = data.get('defect_type', 'Unknown')
                angles = data['angles']
                stresses = data['stresses'][stress_component]
                
                if angles is None or stresses is None:
                    continue
                    
                # Convert to arrays
                try:
                    angles_array = np.array(angles)
                    stresses_array = np.array(stresses)
                except:
                    continue
                
                if len(angles_array) == 0 or len(stresses_array) == 0:
                    continue
                
                # Use enhanced colors
                color = data.get('color', self.enhanced_defect_colors.get(defect_type, 'black'))
                
                # Calculate line width based on eigen strain (thicker for higher strain)
                eigen_strain = data.get('eigen_strain', 1.0)
                line_width = 2 + (eigen_strain * 1.5)
                
                fig.add_trace(go.Scatter(
                    x=angles_array,
                    y=stresses_array,
                    mode='lines+markers',
                    line=dict(color=color, width=line_width),
                    marker=dict(size=8, color=color, symbol='circle'),
                    name=f"{defect_type} (ε*={data.get('eigen_strain', 0):.2f})",
                    hovertemplate='<b>Orientation</b>: %{x:.2f}°<br><b>Stress</b>: %{y:.4f} GPa<extra></extra>',
                    showlegend=True
                ))
        
        # If no traces were added, return empty figure
        if len(fig.data) == 0:
            return self.create_defect_comparison_plot(defect_comparison, stress_component, title)
        
        # Enhanced habit plane marker
        fig.add_vline(x=self.habit_angle, line_width=4, line_dash="dashdot",
                     line_color="green", 
                     annotation=dict(
                         text=f"<b>Habit Plane ({self.habit_angle}°)</b>",
                         font=dict(size=16, color='green', family='Arial Black'),
                         bgcolor="rgba(255, 255, 255, 0.8)",
                         borderpad=4,
                         showarrow=False,
                         yanchor="top",
                         y=0.95
                     ))
        
        # Enhanced system boundaries with labels
        fig.add_vrect(x0=0, x1=5, fillcolor="green", opacity=0.15, line_width=0,
                     annotation=dict(text="<b>System 1</b><br>Perfect Crystal", 
                                    font=dict(size=14, color='darkgreen'),
                                    x=2.5, y=0.05, showarrow=False))
        
        fig.add_vrect(x0=5, x1=20, fillcolor="orange", opacity=0.15, line_width=0,
                     annotation=dict(text="<b>System 2</b><br>SF/Twins", 
                                    font=dict(size=14, color='darkorange'),
                                    x=12.5, y=0.05, showarrow=False))
        
        fig.add_vrect(x0=20, x1=35, fillcolor="red", opacity=0.15, line_width=0,
                     annotation=dict(text="<b>System 3</b><br>Plastic", 
                                    font=dict(size=14, color='darkred'),
                                    x=27.5, y=0.05, showarrow=False))
        
        # Enhanced layout
        fig.update_layout(
            title=dict(
                text=f"{title} - {stress_component.replace('_', ' ').title()}",
                font=dict(size=26, family="Arial Black", color='darkblue'),
                x=0.5,
                y=0.95
            ),
            xaxis=dict(
                title=dict(text='Orientation (°)', 
                          font=dict(size=20, color='black', family='Arial', weight='bold')),
                gridcolor='rgba(100, 100, 100, 0.3)',
                gridwidth=2,
                tickfont=dict(size=16),
                range=[0, 360],
                showline=True,
                linewidth=3,
                linecolor='black',
                mirror=True
            ),
            yaxis=dict(
                title=dict(text=f'{stress_component.replace("_", " ").title()} Stress (GPa)',
                          font=dict(size=20, color='black', family='Arial', weight='bold')),
                gridcolor='rgba(100, 100, 100, 0.3)',
                gridwidth=2,
                tickfont=dict(size=16),
                showline=True,
                linewidth=3,
                linecolor='black',
                mirror=True
            ),
            showlegend=True,
            legend=dict(
                x=1.05,
                y=1.0,
                bgcolor='rgba(255, 255, 255, 0.95)',
                bordercolor='black',
                borderwidth=2,
                font=dict(size=16, family='Arial'),
                itemwidth=40,
                traceorder='normal'
            ),
            width=1200,
            height=700,
            hovermode='x unified',
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(l=100, r=250, t=100, b=100)
        )
        
        return fig

# =============================================
# ENHANCED SINTERING CALCULATOR WITH PUBLICATION PLOTS
# =============================================

class EnhancedSinteringCalculatorWithPublication(EnhancedSinteringCalculator):
    """Enhanced sintering calculator with publication-quality plotting"""
    
    def __init__(self, T0=623.0, beta=0.95, G=30.0, sigma_peak=28.5):
        super().__init__(T0, beta, G, sigma_peak)
        self.pub_visualizer = PublicationChartVisualizer()
    
    def create_publication_sintering_analysis(self, sigma_h, defect_type='Twin', orientation_deg=0.0):
        """Create publication-quality sintering analysis plot"""
        
        # Get detailed analysis
        analysis = self.compute_detailed_sintering_analysis(sigma_h, defect_type, orientation_deg)
        
        # Generate theoretical curves for publication
        max_stress = min(35.0, abs(sigma_h) * 2)
        stresses = np.linspace(0, max_stress, 100)
        
        # Calculate temperatures for different models
        T_exp = self.compute_sintering_temperature_exponential(stresses)
        T_arr_defect = [self.compute_sintering_temperature_arrhenius_defect(s, defect_type) 
                       for s in stresses]
        
        # Create publication-quality figure
        fig = self.pub_visualizer.create_publication_sintering_plot(
            stresses, T_exp, defect_type,
            title=f"Publication: Sintering Analysis for {defect_type}"
        )
        
        return fig, analysis

# =============================================
# ENHANCED MAIN APPLICATION
# =============================================

def main():
    # Configure Streamlit page with enhanced settings
    st.set_page_config(
        page_title="Ag FCC Twin: Habit Plane Vicinity Analysis - Enhanced",
        layout="wide",
        page_icon="🔬",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced CSS for better font control
    st.markdown("""
    <style>
    /* Enhanced font controls */
    .publication-title {
        font-size: 3.2rem !important;
        color: #1E3A8A !important;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        margin-bottom: 1rem;
        font-family: 'Arial Black', sans-serif !important;
    }
    .publication-header {
        font-size: 2.2rem !important;
        color: #374151 !important;
        font-weight: 800 !important;
        border-left: 6px solid #3B82F6;
        padding-left: 1.2rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-family: 'Arial', sans-serif !important;
    }
    .publication-text {
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
        font-family: 'Arial', sans-serif !important;
    }
    .publication-metric {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        font-family: 'Arial', sans-serif !important;
    }
    .chart-container {
        border: 2px solid #E5E7EB;
        border-radius: 10px;
        padding: 20px;
        background-color: white;
        margin-bottom: 20px;
    }
    .publication-controls {
        background-color: #F8FAFC;
        border: 2px solid #3B82F6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .font-slider label {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #1E3A8A !important;
    }
    /* Enhanced table styling */
    .dataframe {
        font-size: 1rem !important;
        font-family: 'Arial', sans-serif !important;
    }
    .dataframe th {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        background-color: #3B82F6 !important;
        color: white !important;
    }
    
    /* Enhanced Plotly chart styling */
    .js-plotly-plot .plotly .main-svg {
        border: 2px solid #E5E7EB !important;
        border-radius: 10px !important;
        background-color: white !important;
    }
    
    /* Larger label sizes for charts */
    .plotly-notifier {
        font-size: 14px !important;
    }
    
    /* Custom scrollbar for better visibility */
    ::-webkit-scrollbar {
        width: 12px !important;
        height: 12px !important;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1 !important;
        border-radius: 10px !important;
    }
    ::-webkit-scrollbar-thumb {
        background: #888 !important;
        border-radius: 10px !important;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Enhanced main header
    st.markdown('<h1 class="publication-title">🔬 Ag FCC Twin: Habit Plane Vicinity Analysis - Publication Quality</h1>', unsafe_allow_html=True)
    
    # Initialize enhanced components
    if 'enhanced_visualizer' not in st.session_state:
        st.session_state.enhanced_visualizer = EnhancedHabitPlaneVisualizer()
    if 'pub_visualizer' not in st.session_state:
        st.session_state.pub_visualizer = PublicationChartVisualizer()
    if 'enhanced_sintering_calc' not in st.session_state:
        st.session_state.enhanced_sintering_calc = EnhancedSinteringCalculatorWithPublication()
    
    # Add publication quality controls to sidebar
    with st.sidebar:
        st.markdown('<div class="publication-controls">', unsafe_allow_html=True)
        st.markdown("#### 🎨 Publication Quality Settings")
        
        # Publication quality toggle
        publication_quality = st.checkbox(
            "Enable Publication Quality",
            value=True,
            help="Enable larger labels, higher DPI, and enhanced visibility for publication-quality figures"
        )
        
        # Font size controls
        if publication_quality:
            st.markdown("##### 📝 Font Size Controls")
            
            col_font1, col_font2 = st.columns(2)
            with col_font1:
                title_size = st.slider(
                    "Title Font Size",
                    min_value=16,
                    max_value=32,
                    value=24,
                    step=1,
                    help="Title font size for publication charts"
                )
            
            with col_font2:
                label_size = st.slider(
                    "Label Font Size",
                    min_value=12,
                    max_value=24,
                    value=18,
                    step=1,
                    help="Axis label font size for publication charts"
                )
            
            # Chart size controls
            st.markdown("##### 📊 Chart Size Controls")
            
            col_size1, col_size2 = st.columns(2)
            with col_size1:
                chart_width = st.slider(
                    "Chart Width",
                    min_value=800,
                    max_value=1600,
                    value=1200,
                    step=100,
                    help="Width of publication charts in pixels"
                )
            
            with col_size2:
                chart_height = st.slider(
                    "Chart Height",
                    min_value=600,
                    max_value=1200,
                    value=800,
                    step=100,
                    help="Height of publication charts in pixels"
                )
            
            # DPI setting
            dpi_setting = st.selectbox(
                "Chart Resolution (DPI)",
                [150, 300, 600],
                index=1,
                help="Higher DPI for better print quality"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize session state for enhanced features
    if 'publication_settings' not in st.session_state:
        st.session_state.publication_settings = {
            'enabled': publication_quality,
            'title_size': title_size if 'title_size' in locals() else 24,
            'label_size': label_size if 'label_size' in locals() else 18,
            'chart_width': chart_width if 'chart_width' in locals() else 1200,
            'chart_height': chart_height if 'chart_height' in locals() else 800,
            'dpi': dpi_setting if 'dpi_setting' in locals() else 300
        }
    
    # Update publication settings
    if publication_quality:
        st.session_state.publication_settings = {
            'enabled': True,
            'title_size': title_size,
            'label_size': label_size,
            'chart_width': chart_width,
            'chart_height': chart_height,
            'dpi': dpi_setting
        }
    
    # Initialize other session state variables (same as original)
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'physics_analyzer' not in st.session_state:
        st.session_state.physics_analyzer = PhysicsBasedStressAnalyzer()
    if 'sintering_calculator' not in st.session_state:
        st.session_state.sintering_calculator = EnhancedSinteringCalculator()
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = PhysicsAwareInterpolator(
            sigma=0.3,
            attention_dim=32,
            num_heads=4,
            attention_blend=0.7,
            use_spatial=True
        )
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = HabitPlaneVisualizer()
    if 'results_manager' not in st.session_state:
        st.session_state.results_manager = EnhancedResultsManager()
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown('<h2 class="physics-header">⚙️ Analysis Configuration</h2>', unsafe_allow_html=True)
        
        # Analysis mode
        analysis_mode = st.radio(
            "Select Analysis Mode:",
            ["Habit Plane Vicinity", "Defect Type Comparison", "Comprehensive Dashboard", "Single Point Analysis"],
            index=0,
            help="Choose the type of analysis to perform"
        )
        
        # Data loading
        st.markdown("#### 📂 Data Management")
        col_load1, col_load2 = st.columns(2)
        with col_load1:
            if st.button("🔄 Load Solutions", use_container_width=True):
                with st.spinner("Loading solutions..."):
                    st.session_state.solutions = st.session_state.loader.load_all_solutions()
                    if st.session_state.solutions:
                        st.success(f"Loaded {len(st.session_state.solutions)} solutions")
                    else:
                        st.warning("No solutions found in directory")
        
        with col_load2:
            if st.button("🧹 Clear Cache", use_container_width=True):
                st.session_state.solutions = []
                st.cache_data.clear()
                st.success("Cache cleared")
        
        # Show loaded solutions info
        if st.session_state.solutions:
            stats = st.session_state.loader.get_solution_statistics(st.session_state.solutions)
            
            with st.expander(f"📊 Loaded Solutions ({stats['total_solutions']})", expanded=False):
                # Defect type distribution
                st.write("**Defect Types:**")
                for defect, count in stats.get('defect_types', {}).items():
                    st.write(f"- {defect}: {count}")
                
                # Orientation statistics
                if 'orientation_stats' in stats:
                    st.write("**Orientation Statistics:**")
                    st.write(f"- Range: {stats['orientation_stats']['min']:.1f}° to {stats['orientation_stats']['max']:.1f}°")
                    st.write(f"- Mean: {stats['orientation_stats']['mean']:.1f}° ± {stats['orientation_stats']['std']:.1f}°")
        
        # Target parameters
        st.markdown("#### 🎯 Target Parameters")
        
        # Defect type with auto eigen strain
        defect_type = st.selectbox(
            "Defect Type",
            ["ISF", "ESF", "Twin", "No Defect"],
            index=2,
            help="Select the defect type for analysis"
        )
        
        # Auto-set eigen strain based on defect type
        eigen_strains = {"ISF": 0.71, "ESF": 1.41, "Twin": 2.12, "No Defect": 0.0}
        default_eps0 = eigen_strains[defect_type]
        
        # Show defect cards
        st.markdown("##### 🔬 Defect Properties")
        col_def1, col_def2 = st.columns(2)
        with col_def1:
            eps0 = st.number_input(
                "Eigen Strain (ε*)",
                min_value=0.0,
                max_value=3.0,
                value=default_eps0,
                step=0.01,
                help="Eigen strain value (auto-set based on defect type)"
            )
        
        with col_def2:
            kappa = st.slider(
                "Interface Energy (κ)",
                min_value=0.1,
                max_value=2.0,
                value=0.6,
                step=0.01,
                help="Interface energy parameter"
            )
        
        shape = st.selectbox(
            "Shape",
            ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle"],
            index=0,
            help="Geometric shape of the defect"
        )
        
        # Region selection
        st.markdown("#### 📍 Analysis Region")
        region_type = st.selectbox(
            "Select Region for Analysis",
            ["bulk", "interface", "defect"],
            index=0,
            help="Material region to analyze: bulk (η<0.4), interface (0.4≤η≤0.6), defect (η>0.6)"
        )
        
        # Vicinity settings
        if analysis_mode == "Habit Plane Vicinity":
            st.markdown("#### 🎯 Vicinity Settings")
            
            vicinity_range = st.slider(
                "Vicinity Range (± degrees)",
                min_value=1.0,
                max_value=45.0,
                value=10.0,
                step=1.0,
                help="Range around habit plane to analyze"
            )
            
            n_points = st.slider(
                "Number of Points",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                help="Number of orientation points in sweep"
            )
        
        # Interpolator settings
        st.markdown("#### 🧠 Interpolation Settings")
        
        col_int1, col_int2 = st.columns(2)
        with col_int1:
            attention_blend = st.slider(
                "Attention Blend",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Blend ratio: attention vs spatial weights (0=spatial only, 1=attention only)"
            )
        
        with col_int2:
            sigma = st.slider(
                "Spatial Sigma",
                min_value=0.05,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Spatial regularization parameter"
            )
        
        use_physics_constraints = st.checkbox(
            "Apply Physics Constraints",
            value=True,
            help="Apply physics-based constraints on interpolation"
        )
        
        # Sintering model settings
        st.markdown("#### 🔥 Sintering Model")
        
        sintering_model = st.radio(
            "Primary Sintering Model",
            ["Arrhenius (Physics-based)", "Exponential (Empirical)", "Both"],
            index=0,
            help="Select primary model for sintering temperature prediction"
        )
        
        # Generate button
        st.markdown("---")
        generate_text = "🚀 Generate Analysis" if analysis_mode != "Comprehensive Dashboard" else "📊 Generate Dashboard"
        if st.button(generate_text, type="primary", use_container_width=True):
            st.session_state.generate_analysis = True
        else:
            st.session_state.generate_analysis = False
    
    # Main content area
    if not st.session_state.solutions:
        st.warning("⚠️ Please load solutions first using the button in the sidebar.")
        
        # Show directory information
        with st.expander("📁 Directory Information", expanded=True):
            st.info(f"**Solutions Directory:** {SOLUTIONS_DIR}")
            st.write("Expected file formats: .pkl, .pickle, .pt, .pth")
            st.write("""
            **Expected data structure:**
            - Each file should contain a dictionary with:
              - 'params': Dictionary of simulation parameters
              - 'history': List of simulation frames
              - Each frame should contain 'eta' (phase field) and 'stresses' (stress fields)
            """)
        
        # Quick start guide
        st.markdown("""
        ## 🚀 Quick Start Guide
        
        1. **Prepare Data**: Place your simulation files in the `numerical_solutions` directory
        2. **Load Solutions**: Click the "Load Solutions" button in the sidebar
        3. **Configure Analysis**: Set your analysis parameters in the sidebar
        4. **Generate Analysis**: Click "Generate Analysis" to start
        
        ## 🔬 Key Features
        
        ### Habit Plane Vicinity Analysis
        - Focus on 54.7° ± specified range
        - Combined attention and Gaussian regularization
        - Physics-aware interpolation with eigen strain constraints
        
        ### Stress Components
        - **Hydrostatic Stress (σ_h)**: Critical for sintering (trace of stress tensor/3)
        - **Von Mises Stress (σ_vm)**: Indicates yield onset
        - **Stress Magnitude (σ_mag)**: Overall stress intensity
        
        ### Sintering Temperature Prediction
        - **Arrhenius Model**: Physics-based with defect-specific parameters
        - **Exponential Model**: Empirical correlation
        - **System Classification**: Maps stress to AgNP sintering systems
        
        ### Visualization
        - Sunburst charts for polar visualization
        - Radar charts for component comparison
        - Interactive plots with Plotly
        - Comprehensive dashboards
        """)
        
    else:
        # Create tabs for different analysis modes
        tab1, tab2, tab3, tab4 = st.tabs([
            "🏠 Overview",
            "📈 Vicinity Analysis",
            "🔬 Defect Comparison",
            "📊 Comprehensive Dashboard"
        ])
        
        with tab1:
            st.markdown('<h2 class="physics-header">🏠 Analysis Overview</h2>', unsafe_allow_html=True)
            
            # Quick statistics
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("Loaded Solutions", len(st.session_state.solutions))
            
            with col_stat2:
                defect_types = set()
                for sol in st.session_state.solutions:
                    params = sol.get('params', {})
                    defect_types.add(params.get('defect_type', 'Unknown'))
                st.metric("Defect Types", len(defect_types))
            
            with col_stat3:
                orientations = []
                for sol in st.session_state.solutions:
                    params = sol.get('params', {})
                    theta = params.get('theta', 0.0)
                    if theta is not None:
                        orientations.append(np.degrees(theta) % 360)
                if orientations:
                    st.metric("Orientation Range", f"{min(orientations):.1f}° - {max(orientations):.1f}°")
                else:
                    st.metric("Orientation Range", "N/A")
            
            with col_stat4:
                has_physics = sum(1 for sol in st.session_state.solutions if sol.get('physics_analysis'))
                st.metric("Physics Analyzed", f"{has_physics}/{len(st.session_state.solutions)}")
            
            # Defect type cards
            st.markdown("#### 🔬 Defect Types with Eigen Strains")
            
            col_def1, col_def2, col_def3, col_def4 = st.columns(4)
            
            defect_info = [
                ("ISF", "Intrinsic Stacking Fault", 0.71, "#FF6B6B"),
                ("ESF", "Extrinsic Stacking Fault", 1.41, "#4ECDC4"),
                ("Twin", "Coherent Twin Boundary", 2.12, "#45B7D1"),
                ("No Defect", "Perfect Crystal", 0.0, "#96CEB4")
            ]
            
            for i, (name, desc, strain, color) in enumerate(defect_info):
                with [col_def1, col_def2, col_def3, col_def4][i]:
                    st.markdown(f"""
                    <div class="defect-card {'isf' if name == 'ISF' else 'esf' if name == 'ESF' else 'twin' if name == 'Twin' else 'perfect'}-card">
                    <div style="font-size: 1.5rem; font-weight: bold; color: {color};">{name}</div>
                    <div style="font-size: 0.9rem; color: #666; margin: 0.5rem 0;">{desc}</div>
                    <div style="font-size: 1.2rem; font-weight: bold; color: #333;">ε* = {strain}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # System classification
            st.markdown("#### 🏷️ System Classification")
            
            col_sys1, col_sys2, col_sys3 = st.columns(3)
            
            system_info = [
                ("System 1", "Perfect Crystal", "σ < 5 GPa", "620-630 K", "#10B981"),
                ("System 2", "SF/Twins", "5 ≤ σ < 20 GPa", "450-550 K", "#F59E0B"),
                ("System 3", "Plastic Deformation", "σ ≥ 20 GPa", "350-400 K", "#EF4444")
            ]
            
            for i, (name, desc, stress_range, temp_range, color) in enumerate(system_info):
                with [col_sys1, col_sys2, col_sys3][i]:
                    st.markdown(f"""
                    <div class="system-metric" style="background-color: {color};">
                    <div style="font-size: 1.2rem; font-weight: bold;">{name}</div>
                    <div style="font-size: 0.9rem; opacity: 0.9;">{desc}</div>
                    <div style="font-size: 0.8rem; margin-top: 0.5rem;">{stress_range}</div>
                    <div style="font-size: 0.8rem;">{temp_range}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Quick analysis buttons
            st.markdown("#### ⚡ Quick Analysis")
            
            col_q1, col_q2, col_q3 = st.columns(3)
            
            with col_q1:
                if st.button("🎯 Analyze Habit Plane", use_container_width=True):
                    st.session_state.quick_analysis = "habit_plane"
            
            with col_q2:
                if st.button("🔬 Compare Defects", use_container_width=True):
                    st.session_state.quick_analysis = "defect_compare"
            
            with col_q3:
                if st.button("📊 Generate Dashboard", use_container_width=True):
                    st.session_state.quick_analysis = "dashboard"
        
        with tab2:
            st.markdown('<h2 class="physics-header">📈 Habit Plane Vicinity Analysis</h2>', unsafe_allow_html=True)
            
            if st.session_state.get('generate_analysis', False) or st.session_state.get('quick_analysis') == 'habit_plane':
                
                with st.spinner("Performing vicinity analysis..."):
                    # Prepare target parameters
                    target_params = {
                        'defect_type': defect_type,
                        'shape': shape,
                        'eps0': eps0,
                        'kappa': kappa
                    }
                    
                    # Prepare analysis parameters
                    analysis_params = {
                        'vicinity_range': vicinity_range if 'vicinity_range' in locals() else 10.0,
                        'n_points': n_points if 'n_points' in locals() else 50,
                        'region_type': region_type,
                        'attention_blend': attention_blend,
                        'sigma': sigma,
                        'use_physics_constraints': use_physics_constraints
                    }
                    
                    # Update interpolator settings
                    st.session_state.interpolator.attention_blend = attention_blend
                    st.session_state.interpolator.sigma = sigma
                    
                    # Perform vicinity sweep
                    vicinity_sweep = st.session_state.interpolator.create_vicinity_sweep(
                        st.session_state.solutions,
                        target_params,
                        vicinity_range=analysis_params['vicinity_range'],
                        n_points=analysis_params['n_points'],
                        region_type=analysis_params['region_type']
                    )
                    
                    if vicinity_sweep:
                        st.success(f"✅ Generated vicinity sweep with {analysis_params['n_points']} points")
                        
                        # Store in session state
                        st.session_state.vicinity_sweep = vicinity_sweep
                        st.session_state.current_target_params = target_params
                        st.session_state.current_analysis_params = analysis_params
                        
                        # Display results
                        st.markdown("#### 📊 Analysis Results")
                        
                        # Create columns for metrics
                        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                        
                        # Habit plane stress
                        angles = np.array(vicinity_sweep['angles'])
                        habit_angle = 54.7
                        habit_idx = np.argmin(np.abs(angles - habit_angle))
                        
                        with col_res1:
                            sigma_hydro = vicinity_sweep['stresses']['sigma_hydro'][habit_idx]
                            st.metric(
                                "Habit Plane σ_h",
                                f"{sigma_hydro:.3f} GPa",
                                "Hydrostatic Stress"
                            )
                        
                        with col_res2:
                            von_mises = vicinity_sweep['stresses']['von_mises'][habit_idx]
                            st.metric(
                                "Habit Plane σ_vm",
                                f"{von_mises:.3f} GPa",
                                "Von Mises Stress"
                            )
                        
                        with col_res3:
                            T_sinter = vicinity_sweep['sintering_temps']['arrhenius_defect'][habit_idx]
                            st.metric(
                                "Habit Plane T_sinter",
                                f"{T_sinter:.1f} K",
                                f"{T_sinter-273.15:.1f}°C"
                            )
                        
                        with col_res4:
                            system_info = st.session_state.sintering_calculator.map_system_to_temperature(sigma_hydro)
                            st.metric(
                                "System Classification",
                                system_info[0].split('(')[0].strip(),
                                system_info[1][1] - system_info[1][0]
                            )
                        
                        # Create visualizations
                        st.markdown("#### 📈 Visualizations")
                        
                        # Choose visualizer based on publication quality setting
                        if st.session_state.publication_settings['enabled']:
                            visualizer = st.session_state.enhanced_visualizer
                        else:
                            visualizer = st.session_state.visualizer
                        
                        # Sunburst for hydrostatic stress
                        if st.session_state.publication_settings['enabled']:
                            fig_sunburst = st.session_state.enhanced_visualizer.create_enhanced_vicinity_sunburst(
                                vicinity_sweep['angles'],
                                vicinity_sweep['stresses']['sigma_hydro'],
                                stress_component='sigma_hydro',
                                title=f"Habit Plane Vicinity - {defect_type}",
                                publication_quality=True
                            )
                        else:
                            fig_sunburst = st.session_state.visualizer.create_vicinity_sunburst(
                                vicinity_sweep['angles'],
                                vicinity_sweep['stresses']['sigma_hydro'],
                                stress_component='sigma_hydro',
                                title=f"Habit Plane Vicinity - {defect_type}"
                            )
                        
                        if fig_sunburst:
                            st.plotly_chart(fig_sunburst, use_container_width=True)
                        
                        # Line plots for all stress components
                        if st.session_state.publication_settings['enabled']:
                            # Publication quality plot
                            fig_line, ax_line = plt.subplots(figsize=(14, 8), dpi=st.session_state.publication_settings['dpi'])
                            
                            # Set publication parameters
                            st.session_state.pub_visualizer.publication_params['title_font_size'] = st.session_state.publication_settings['title_size']
                            st.session_state.pub_visualizer.publication_params['axis_label_font_size'] = st.session_state.publication_settings['label_size']
                            st.session_state.pub_visualizer.publication_params['tick_label_font_size'] = st.session_state.publication_settings['label_size'] - 2
                        else:
                            fig_line, ax_line = plt.subplots(figsize=(12, 6))
                        
                        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
                        line_width = 3 if st.session_state.publication_settings['enabled'] else 2
                        marker_size = 8 if st.session_state.publication_settings['enabled'] else 6
                        
                        for idx, (comp, color) in enumerate(zip(['sigma_hydro', 'von_mises', 'sigma_mag'], colors)):
                            ax_line.plot(
                                vicinity_sweep['angles'],
                                vicinity_sweep['stresses'][comp],
                                color=color,
                                linewidth=line_width,
                                marker='o' if st.session_state.publication_settings['enabled'] else None,
                                markersize=marker_size,
                                label=comp.replace('_', ' ').title()
                            )
                        
                        ax_line.axvline(habit_angle, color='green', linestyle='--', linewidth=line_width,
                                      label=f'Habit Plane ({habit_angle}°)')
                        ax_line.set_xlabel('Orientation (°)', fontsize=st.session_state.publication_settings['label_size'] if st.session_state.publication_settings['enabled'] else 12)
                        ax_line.set_ylabel('Stress (GPa)', fontsize=st.session_state.publication_settings['label_size'] if st.session_state.publication_settings['enabled'] else 12)
                        
                        if st.session_state.publication_settings['enabled']:
                            ax_line.set_title('Stress Components in Habit Plane Vicinity', 
                                            fontsize=st.session_state.publication_settings['title_size'], 
                                            fontweight='bold')
                        else:
                            ax_line.set_title('Stress Components in Habit Plane Vicinity', fontsize=14, fontweight='bold')
                        
                        ax_line.legend(fontsize=st.session_state.publication_settings['label_size'] - 2 if st.session_state.publication_settings['enabled'] else 11)
                        ax_line.grid(True, alpha=0.3)
                        
                        col_viz1, col_viz2 = st.columns(2)
                        with col_viz1:
                            st.pyplot(fig_line)
                            plt.close(fig_line)
                        
                        # Sintering temperature plot
                        with col_viz2:
                            if st.session_state.publication_settings['enabled']:
                                # Publication quality plot
                                fig_temp, ax_temp = plt.subplots(figsize=(12, 7), dpi=st.session_state.publication_settings['dpi'])
                            else:
                                fig_temp, ax_temp = plt.subplots(figsize=(10, 6))
                            
                            ax_temp.plot(
                                vicinity_sweep['angles'],
                                vicinity_sweep['sintering_temps']['exponential'],
                                color='red',
                                linewidth=line_width,
                                marker='s' if st.session_state.publication_settings['enabled'] else None,
                                markersize=marker_size,
                                label='Exponential Model'
                            )
                            
                            ax_temp.plot(
                                vicinity_sweep['angles'],
                                vicinity_sweep['sintering_temps']['arrhenius_defect'],
                                color='blue',
                                linewidth=line_width,
                                linestyle='--',
                                marker='^' if st.session_state.publication_settings['enabled'] else None,
                                markersize=marker_size,
                                label='Arrhenius Model (Defect)'
                            )
                            
                            ax_temp.axvline(habit_angle, color='green', linestyle='--', linewidth=line_width)
                            ax_temp.set_xlabel('Orientation (°)', fontsize=st.session_state.publication_settings['label_size'] if st.session_state.publication_settings['enabled'] else 12)
                            ax_temp.set_ylabel('Sintering Temperature (K)', fontsize=st.session_state.publication_settings['label_size'] if st.session_state.publication_settings['enabled'] else 12)
                            
                            if st.session_state.publication_settings['enabled']:
                                ax_temp.set_title('Sintering Temperature Prediction', 
                                                fontsize=st.session_state.publication_settings['title_size'], 
                                                fontweight='bold')
                            else:
                                ax_temp.set_title('Sintering Temperature Prediction', fontsize=14, fontweight='bold')
                            
                            ax_temp.legend(fontsize=st.session_state.publication_settings['label_size'] - 2 if st.session_state.publication_settings['enabled'] else 11)
                            ax_temp.grid(True, alpha=0.3)
                            
                            # Add Celsius on secondary axis
                            ax_temp2 = ax_temp.twinx()
                            celsius_ticks = ax_temp.get_yticks()
                            ax_temp2.set_ylim(ax_temp.get_ylim())
                            ax_temp2.set_yticklabels([f'{t-273.15:.0f}°C' for t in celsius_ticks])
                            ax_temp2.set_ylabel('Temperature (°C)', fontsize=st.session_state.publication_settings['label_size'] if st.session_state.publication_settings['enabled'] else 12)
                            
                            st.pyplot(fig_temp)
                            plt.close(fig_temp)
                        
                        # Publication quality sintering analysis
                        if st.session_state.publication_settings['enabled']:
                            st.markdown("#### 📊 Publication-Quality Sintering Analysis")
                            
                            # Generate publication-quality sintering plot
                            max_stress = max(vicinity_sweep['stresses']['sigma_hydro'])
                            stresses = np.linspace(0, max_stress, 100)
                            T_exp = [st.session_state.sintering_calculator.compute_sintering_temperature_exponential(s) for s in stresses]
                            
                            fig_pub = st.session_state.pub_visualizer.create_publication_sintering_plot(
                                stresses, T_exp, defect_type,
                                title=f"Publication: Sintering Analysis for {defect_type}"
                            )
                            
                            st.pyplot(fig_pub)
                            plt.close(fig_pub)
                        
                        # Export options
                        st.markdown("#### 📤 Export Results")
                        
                        col_exp1, col_exp2, col_exp3 = st.columns(3)
                        
                        with col_exp1:
                            if st.button("💾 Export JSON", use_container_width=True):
                                # Prepare report
                                report = st.session_state.results_manager.prepare_vicinity_analysis_report(
                                    vicinity_sweep,
                                    {},
                                    target_params,
                                    analysis_params
                                )
                                
                                json_str = json.dumps(report, indent=2, default=st.session_state.results_manager._json_serializer)
                                
                                st.download_button(
                                    label="📥 Download JSON",
                                    data=json_str,
                                    file_name=f"vicinity_analysis_{defect_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json",
                                    use_container_width=True
                                )
                        
                        with col_exp2:
                            if st.button("📊 Export CSV", use_container_width=True):
                                # Create CSV data
                                rows = []
                                angles = vicinity_sweep['angles']
                                
                                for i in range(len(angles)):
                                    row = {
                                        'angle_deg': angles[i],
                                        'sigma_hydro_gpa': vicinity_sweep['stresses']['sigma_hydro'][i],
                                        'von_mises_gpa': vicinity_sweep['stresses']['von_mises'][i],
                                        'sigma_mag_gpa': vicinity_sweep['stresses']['sigma_mag'][i],
                                        'T_sinter_exponential_k': vicinity_sweep['sintering_temps']['exponential'][i],
                                        'T_sinter_arrhenius_k': vicinity_sweep['sintering_temps']['arrhenius_defect'][i]
                                    }
                                    rows.append(row)
                                
                                df = pd.DataFrame(rows)
                                csv = df.to_csv(index=False)
                                
                                st.download_button(
                                    label="📥 Download CSV",
                                    data=csv,
                                    file_name=f"vicinity_data_{defect_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                        
                        with col_exp3:
                            if st.button("📦 Export Complete Package", use_container_width=True):
                                report = st.session_state.results_manager.prepare_vicinity_analysis_report(
                                    vicinity_sweep,
                                    {},
                                    target_params,
                                    analysis_params
                                )
                                
                                zip_buffer = st.session_state.results_manager.create_comprehensive_export(report)
                                
                                st.download_button(
                                    label="📥 Download ZIP",
                                    data=zip_buffer.getvalue(),
                                    file_name=f"vicinity_analysis_package_{defect_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                    mime="application/zip",
                                    use_container_width=True
                                )
                    
                    else:
                        st.error("Failed to generate vicinity sweep. Please check your data and parameters.")
            
            else:
                st.info("👈 Configure analysis parameters in the sidebar and click 'Generate Analysis'")
                
                # Show example visualization
                st.markdown("#### 📊 Example Visualization")
                
                # Create example data
                example_angles = np.linspace(44.7, 64.7, 50)
                example_stress = 20 * np.exp(-(example_angles - 54.7)**2 / (2*5**2)) + 5
                example_temp = 623 * np.exp(-example_stress / 30) + 50 * np.sin(np.radians(example_angles))
                
                if st.session_state.publication_settings['enabled']:
                    fig_example, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=st.session_state.publication_settings['dpi'])
                else:
                    fig_example, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                # Stress plot
                ax1.plot(example_angles, example_stress, 'b-', linewidth=3 if st.session_state.publication_settings['enabled'] else 2)
                ax1.axvline(54.7, color='green', linestyle='--', linewidth=2, label='Habit Plane (54.7°)')
                ax1.fill_between(example_angles, example_stress, alpha=0.2, color='blue')
                ax1.set_xlabel('Orientation (°)', fontsize=st.session_state.publication_settings['label_size'] if st.session_state.publication_settings['enabled'] else 12)
                ax1.set_ylabel('Hydrostatic Stress (GPa)', fontsize=st.session_state.publication_settings['label_size'] if st.session_state.publication_settings['enabled'] else 12)
                
                if st.session_state.publication_settings['enabled']:
                    ax1.set_title('Example: Stress Concentration at Habit Plane', 
                                fontsize=st.session_state.publication_settings['title_size'], 
                                fontweight='bold')
                else:
                    ax1.set_title('Example: Stress Concentration at Habit Plane', fontsize=14, fontweight='bold')
                
                ax1.legend(fontsize=st.session_state.publication_settings['label_size'] - 2 if st.session_state.publication_settings['enabled'] else 11)
                ax1.grid(True, alpha=0.3)
                
                # Temperature plot
                ax2.plot(example_angles, example_temp, 'r-', linewidth=3 if st.session_state.publication_settings['enabled'] else 2)
                ax2.axvline(54.7, color='green', linestyle='--', linewidth=2, label='Habit Plane (54.7°)')
                ax2.fill_between(example_angles, example_temp, alpha=0.2, color='red')
                ax2.set_xlabel('Orientation (°)', fontsize=st.session_state.publication_settings['label_size'] if st.session_state.publication_settings['enabled'] else 12)
                ax2.set_ylabel('Sintering Temperature (K)', fontsize=st.session_state.publication_settings['label_size'] if st.session_state.publication_settings['enabled'] else 12)
                
                if st.session_state.publication_settings['enabled']:
                    ax2.set_title('Example: Temperature Reduction at Habit Plane', 
                                fontsize=st.session_state.publication_settings['title_size'], 
                                fontweight='bold')
                else:
                    ax2.set_title('Example: Temperature Reduction at Habit Plane', fontsize=14, fontweight='bold')
                
                ax2.legend(fontsize=st.session_state.publication_settings['label_size'] - 2 if st.session_state.publication_settings['enabled'] else 11)
                ax2.grid(True, alpha=0.3)
                
                # Add Celsius on secondary axis
                ax2_2 = ax2.twinx()
                celsius_ticks = ax2.get_yticks()
                ax2_2.set_ylim(ax2.get_ylim())
                ax2_2.set_yticklabels([f'{t-273.15:.0f}°C' for t in celsius_ticks])
                ax2_2.set_ylabel('Temperature (°C)', fontsize=st.session_state.publication_settings['label_size'] if st.session_state.publication_settings['enabled'] else 12)
                
                st.pyplot(fig_example)
                plt.close(fig_example)
        
        with tab3:
            st.markdown('<h2 class="physics-header">🔬 Defect Type Comparison</h2>', unsafe_allow_html=True)
            
            if st.session_state.get('generate_analysis', False) or st.session_state.get('quick_analysis') == 'defect_compare':
                
                with st.spinner("Comparing defect types..."):
                    # Compare all defect types
                    defect_comparison = st.session_state.interpolator.compare_defect_types(
                        st.session_state.solutions,
                        angle_range=(0, 360),
                        n_points=100,
                        region_type=region_type,
                        shapes=[shape]
                    )
                    
                    if defect_comparison:
                        st.success(f"✅ Generated comparison of {len(defect_comparison)} defect types")
                        
                        # Store in session state
                        st.session_state.defect_comparison = defect_comparison
                        
                        # Display defect comparison
                        st.markdown("#### 📊 Defect Comparison Results")
                        
                        # Create tabs for different views
                        comp_tab1, comp_tab2, comp_tab3 = st.tabs(["Stress Comparison", "Sintering Comparison", "Radar View"])
                        
                        with comp_tab1:
                            # Stress comparison plot
                            if st.session_state.publication_settings['enabled']:
                                fig_comp = st.session_state.enhanced_visualizer.create_enhanced_defect_comparison_plot(
                                    defect_comparison,
                                    stress_component='sigma_hydro',
                                    title="Hydrostatic Stress Comparison",
                                    publication_quality=True
                                )
                            else:
                                fig_comp = st.session_state.visualizer.create_defect_comparison_plot(
                                    defect_comparison,
                                    stress_component='sigma_hydro',
                                    title="Hydrostatic Stress Comparison"
                                )
                            
                            if fig_comp:
                                st.plotly_chart(fig_comp, use_container_width=True)
                        
                        with comp_tab2:
                            # Sintering temperature comparison
                            if st.session_state.publication_settings['enabled']:
                                fig_sinter_comp, ax_sinter = plt.subplots(figsize=(14, 8), dpi=st.session_state.publication_settings['dpi'])
                            else:
                                fig_sinter_comp, ax_sinter = plt.subplots(figsize=(12, 6))
                            
                            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                            line_width = 3 if st.session_state.publication_settings['enabled'] else 2
                            marker_size = 8 if st.session_state.publication_settings['enabled'] else 6
                            
                            for idx, (key, data) in enumerate(defect_comparison.items()):
                                if idx < len(colors):
                                    ax_sinter.plot(
                                        data['angles'],
                                        data['sintering_temps'],
                                        color=colors[idx],
                                        linewidth=line_width,
                                        marker='o' if st.session_state.publication_settings['enabled'] else None,
                                        markersize=marker_size,
                                        label=f"{data['defect_type']} (ε*={data['eigen_strain']})"
                                    )
                            
                            ax_sinter.axvline(54.7, color='green', linestyle='--', linewidth=line_width,
                                            label='Habit Plane (54.7°)')
                            ax_sinter.set_xlabel('Orientation (°)', fontsize=st.session_state.publication_settings['label_size'] if st.session_state.publication_settings['enabled'] else 12)
                            ax_sinter.set_ylabel('Sintering Temperature (K)', fontsize=st.session_state.publication_settings['label_size'] if st.session_state.publication_settings['enabled'] else 12)
                            
                            if st.session_state.publication_settings['enabled']:
                                ax_sinter.set_title('Sintering Temperature Comparison by Defect Type', 
                                                  fontsize=st.session_state.publication_settings['title_size'], 
                                                  fontweight='bold')
                            else:
                                ax_sinter.set_title('Sintering Temperature Comparison by Defect Type', fontsize=14, fontweight='bold')
                            
                            ax_sinter.legend(fontsize=st.session_state.publication_settings['label_size'] - 2 if st.session_state.publication_settings['enabled'] else 10)
                            ax_sinter.grid(True, alpha=0.3)
                            
                            # Add Celsius on secondary axis
                            ax_sinter2 = ax_sinter.twinx()
                            celsius_ticks = ax_sinter.get_yticks()
                            ax_sinter2.set_ylim(ax_sinter.get_ylim())
                            ax_sinter2.set_yticklabels([f'{t-273.15:.0f}°C' for t in celsius_ticks])
                            ax_sinter2.set_ylabel('Temperature (°C)', fontsize=st.session_state.publication_settings['label_size'] if st.session_state.publication_settings['enabled'] else 12)
                            
                            st.pyplot(fig_sinter_comp)
                            plt.close(fig_sinter_comp)
                        
                        with comp_tab3:
                            # Radar comparison for habit plane vicinity
                            habit_range = 30.0
                            min_angle = 54.7 - habit_range
                            max_angle = 54.7 + habit_range
                            
                            # Filter data for habit plane vicinity
                            vicinity_comparison = {}
                            for key, data in defect_comparison.items():
                                angles = np.array(data['angles'])
                                mask = (angles >= min_angle) & (angles <= max_angle)
                                
                                if np.any(mask):
                                    vicinity_comparison[key] = {
                                        'angles': angles[mask].tolist(),
                                        'stresses': {comp: np.array(vals)[mask].tolist() 
                                                   for comp, vals in data['stresses'].items()},
                                        'defect_type': data['defect_type'],
                                        'color': data['color']
                                    }
                            
                            # Create radar comparison
                            if st.session_state.publication_settings['enabled']:
                                # Use publication-quality radar chart
                                stress_components = {}
                                for key, data in vicinity_comparison.items():
                                    if 'sigma_hydro' in data['stresses']:
                                        stress_components[data['defect_type']] = data['stresses']['sigma_hydro']
                                
                                if stress_components:
                                    fig_radar = st.session_state.pub_visualizer.create_publication_radar_chart(
                                        vicinity_comparison[list(vicinity_comparison.keys())[0]]['angles'],
                                        stress_components,
                                        title="Stress Components in Habit Plane Vicinity"
                                    )
                                    st.pyplot(fig_radar)
                                    plt.close(fig_radar)
                            else:
                                # Use standard radar chart
                                fig_radar = st.session_state.visualizer.create_stress_comparison_radar(
                                    vicinity_comparison,
                                    title="Stress Components in Habit Plane Vicinity"
                                )
                                
                                if fig_radar:
                                    st.plotly_chart(fig_radar, use_container_width=True)
                        
                        # Summary statistics
                        st.markdown("#### 📈 Summary Statistics")
                        
                        # Calculate statistics for each defect
                        summary_data = []
                        for key, data in defect_comparison.items():
                            if 'stresses' in data and 'sigma_hydro' in data['stresses']:
                                stresses = data['stresses']['sigma_hydro']
                                if stresses:
                                    summary_data.append({
                                        'Defect Type': data['defect_type'],
                                        'Eigen Strain': data['eigen_strain'],
                                        'Max Stress (GPa)': f"{max(stresses):.3f}",
                                        'Mean Stress (GPa)': f"{np.mean(stresses):.3f}",
                                        'Stress Range (GPA)': f"{max(stresses) - min(stresses):.3f}",
                                        'Min T_sinter (K)': f"{min(data['sintering_temps']):.1f}",
                                        'Max T_sinter (K)': f"{max(data['sintering_temps']):.1f}"
                                    })
                        
                        if summary_data:
                            df_summary = pd.DataFrame(summary_data)
                            st.dataframe(df_summary, use_container_width=True)
                        
                        # Export comparison data
                        st.markdown("#### 📤 Export Comparison Data")
                        
                        col_exp1, col_exp2 = st.columns(2)
                        
                        with col_exp1:
                            # JSON export
                            comparison_json = json.dumps(
                                st.session_state.defect_comparison,
                                indent=2,
                                default=st.session_state.results_manager._json_serializer
                            )
                            
                            st.download_button(
                                label="💾 Export JSON",
                                data=comparison_json,
                                file_name=f"defect_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                        
                        with col_exp2:
                            # CSV export
                            rows = []
                            for key, data in defect_comparison.items():
                                angles = data['angles']
                                for i in range(len(angles)):
                                    row = {
                                        'defect_type': data['defect_type'],
                                        'eigen_strain': data['eigen_strain'],
                                        'angle_deg': angles[i]
                                    }
                                    
                                    for comp, stresses in data['stresses'].items():
                                        if i < len(stresses):
                                            row[f'{comp}_gpa'] = stresses[i]
                                    
                                    if i < len(data['sintering_temps']):
                                        row['T_sinter_k'] = data['sintering_temps'][i]
                                    
                                    rows.append(row)
                            
                            if rows:
                                df = pd.DataFrame(rows)
                                csv = df.to_csv(index=False)
                                
                                st.download_button(
                                    label="📊 Export CSV",
                                    data=csv,
                                    file_name=f"defect_comparison_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                    
                    else:
                        st.error("Failed to generate defect comparison. Please check your data.")
            
            else:
                st.info("👈 Configure analysis parameters in the sidebar and click 'Generate Analysis'")
                
                # Show defect comparison info
                st.markdown("""
                #### 🔬 Defect Comparison Analysis
                
                This analysis compares different defect types (ISF, ESF, Twin, No Defect) across all orientations.
                
                **Key comparisons:**
                1. **Stress distribution** - How each defect concentrates stress
                2. **Sintering temperature** - Temperature reduction capability
                3. **Habit plane effects** - Special behavior at 54.7°
                
                **Expected insights:**
                - Twin boundaries show maximum stress concentration
                - ISF/ESF have intermediate effects
                - Perfect crystals have minimal stress
                - Habit plane shows peak effects for twins
                """)
        
        with tab4:
            st.markdown('<h2 class="physics-header">📊 Comprehensive Dashboard</h2>', unsafe_allow_html=True)
            
            if st.session_state.get('generate_analysis', False) or st.session_state.get('quick_analysis') == 'dashboard':
                
                with st.spinner("Generating comprehensive dashboard..."):
                    # Check if we have both vicinity sweep and defect comparison
                    if ('vicinity_sweep' not in st.session_state or 
                        'defect_comparison' not in st.session_state):
                        
                        st.warning("Please run both Vicinity Analysis and Defect Comparison first.")
                        
                        col_run1, col_run2 = st.columns(2)
                        with col_run1:
                            if st.button("🏃 Run Vicinity Analysis", use_container_width=True):
                                st.session_state.quick_analysis = "habit_plane"
                                st.rerun()
                        
                        with col_run2:
                            if st.button("🏃 Run Defect Comparison", use_container_width=True):
                                st.session_state.quick_analysis = "defect_compare"
                                st.rerun()
                    
                    else:
                        # Generate comprehensive dashboard
                        vicinity_sweep = st.session_state.vicinity_sweep
                        defect_comparison = st.session_state.defect_comparison
                        
                        # Create comprehensive visualization
                        fig_dashboard = st.session_state.visualizer.create_comprehensive_dashboard(
                            vicinity_sweep,
                            defect_comparison,
                            title=f"Comprehensive Analysis - {st.session_state.current_target_params.get('defect_type', 'Unknown')}"
                        )
                        
                        if fig_dashboard:
                            # Adjust figure size for publication quality
                            if st.session_state.publication_settings['enabled']:
                                fig_dashboard.update_layout(
                                    width=st.session_state.publication_settings['chart_width'],
                                    height=st.session_state.publication_settings['chart_height']
                                )
                            
                            st.plotly_chart(fig_dashboard, use_container_width=True)
                        
                        # Additional analysis
                        st.markdown("#### 📈 Advanced Analysis")
                        
                        # Create tabs for different analyses
                        adv_tab1, adv_tab2, adv_tab3 = st.tabs(["Physics Analysis", "Sintering Optimization", "Export Package"])
                        
                        with adv_tab1:
                            # Physics-based analysis
                            st.markdown("##### 🔬 Physics-Based Analysis")
                            
                            # Calculate stress intensity factors
                            st.write("**Stress Intensity Factors (K):**")
                            
                            col_k1, col_k2, col_k3, col_k4 = st.columns(4)
                            
                            defect_types = ['ISF', 'ESF', 'Twin', 'No Defect']
                            for i, defect in enumerate(defect_types):
                                with [col_k1, col_k2, col_k3, col_k4][i]:
                                    # Find max stress for this defect
                                    max_stress = 0
                                    for key, data in defect_comparison.items():
                                        if data['defect_type'] == defect:
                                            if 'sigma_hydro' in data['stresses']:
                                                max_stress = max(data['stresses']['sigma_hydro'])
                                                break
                                    
                                    # Calculate K
                                    K = st.session_state.physics_analyzer.compute_stress_intensity_factor(
                                        {'sigma_hydro': {'max_abs': max_stress}},
                                        st.session_state.physics_analyzer.get_eigen_strain(defect),
                                        defect
                                    )
                                    
                                    st.metric(
                                        f"K for {defect}",
                                        f"{K:.2f} MPa√m",
                                        "Stress Intensity"
                                    )
                            
                            # Crystal orientation analysis
                            st.markdown("##### 🧊 Crystal Orientation Effects")
                            
                            orientation_effects = []
                            for angle in [0, 30, 45, 54.7, 60, 90]:
                                effect = st.session_state.physics_analyzer.analyze_crystal_orientation_effects(
                                    {},  # Empty stress data for basic analysis
                                    angle
                                )
                                orientation_effects.append(effect)
                            
                            if orientation_effects:
                                df_orientation = pd.DataFrame(orientation_effects)
                                st.dataframe(df_orientation, use_container_width=True)
                        
                        with adv_tab2:
                            # Sintering optimization
                            st.markdown("##### 🔥 Sintering Optimization Analysis")
                            
                            # Find optimal orientation for each defect
                            optimal_data = []
                            for key, data in defect_comparison.items():
                                if 'angles' in data and 'sintering_temps' in data:
                                    temps = data['sintering_temps']
                                    angles = data['angles']
                                    
                                    # Find minimum sintering temperature
                                    min_temp_idx = np.argmin(temps)
                                    min_temp = temps[min_temp_idx]
                                    opt_angle = angles[min_temp_idx]
                                    
                                    optimal_data.append({
                                        'Defect Type': data['defect_type'],
                                        'Optimal Angle (°)': f"{opt_angle:.1f}",
                                        'Min T_sinter (K)': f"{min_temp:.1f}",
                                        'Min T_sinter (°C)': f"{min_temp-273.15:.1f}",
                                        'Temperature Reduction (K)': f"{623.0 - min_temp:.1f}",
                                        'Is Near Habit Plane': abs(opt_angle - 54.7) < 5.0
                                    })
                            
                            if optimal_data:
                                df_optimal = pd.DataFrame(optimal_data)
                                st.dataframe(df_optimal, use_container_width=True)
                                
                                # Recommendation
                                st.markdown("##### 💡 Optimization Recommendation")
                                
                                best_defect = min(optimal_data, key=lambda x: float(x['Min T_sinter (K)'].split()[0]))
                                
                                st.info(f"""
                                **Recommended Configuration:**
                                - **Defect Type:** {best_defect['Defect Type']}
                                - **Optimal Orientation:** {best_defect['Optimal Angle (°)']}°
                                - **Minimum Sintering Temperature:** {best_defect['Min T_sinter (K)']} K ({best_defect['Min T_sinter (°C)']}°C)
                                - **Temperature Reduction:** {best_defect['Temperature Reduction (K)']} K from reference
                                
                                **Note:** {best_defect['Defect Type']} provides the lowest sintering temperature
                                among all analyzed defect types.
                                """)
                        
                        with adv_tab3:
                            # Comprehensive export
                            st.markdown("##### 📦 Comprehensive Export Package")
                            
                            st.write("""
                            The comprehensive export package includes:
                            1. Complete JSON report with all analysis data
                            2. CSV files for all datasets
                            3. README with analysis documentation
                            4. Python script for data processing
                            5. Configuration file
                            """)
                            
                            # Prepare comprehensive report
                            if st.button("🛠️ Prepare Comprehensive Report", use_container_width=True):
                                with st.spinner("Preparing comprehensive report..."):
                                    report = st.session_state.results_manager.prepare_vicinity_analysis_report(
                                        vicinity_sweep,
                                        defect_comparison,
                                        st.session_state.current_target_params,
                                        st.session_state.current_analysis_params
                                    )
                                    
                                    zip_buffer = st.session_state.results_manager.create_comprehensive_export(report)
                                    
                                    st.download_button(
                                        label="📥 Download Complete Package",
                                        data=zip_buffer.getvalue(),
                                        file_name=f"comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                        mime="application/zip",
                                        use_container_width=True
                                    )
            
            else:
                st.info("👈 Configure analysis parameters in the sidebar and click 'Generate Analysis'")
                
                # Dashboard features
                st.markdown("""
                #### 📊 Comprehensive Dashboard Features
                
                The comprehensive dashboard provides an integrated view of all analysis results:
                
                **1. Multi-Panel Visualization**
                - Sunburst charts for polar stress visualization
                - Line plots for detailed orientation dependence
                - Radar charts for component comparison
                - Defect comparison across all types
                
                **2. Advanced Analysis**
                - Physics-based stress intensity calculations
                - Crystal orientation effects
                - Sintering temperature optimization
                - System classification mapping
                
                **3. Comprehensive Export**
                - Complete data package with all results
                - Processing scripts for further analysis
                - Documentation and configuration files
                
                **To generate the dashboard:**
                1. Run both Vicinity Analysis and Defect Comparison
                2. Click "Generate Dashboard" in the sidebar
                3. Explore the comprehensive results
                """)
                
                # Quick status check
                if 'vicinity_sweep' in st.session_state:
                    st.success("✅ Vicinity analysis data available")
                else:
                    st.warning("⚠️ Vicinity analysis not yet run")
                
                if 'defect_comparison' in st.session_state:
                    st.success("✅ Defect comparison data available")
                else:
                    st.warning("⚠️ Defect comparison not yet run")

# =============================================
# RUN THE ENHANCED APPLICATION
# =============================================
if __name__ == "__main__":
    main()
