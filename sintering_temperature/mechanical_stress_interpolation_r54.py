# =============================================
# DOMAIN PARAMETERS (NEW SECTION)
# =============================================
class SimulationDomain:
    """Physical domain specification for phase-field simulations"""
    N_POINTS = 128          # Grid resolution (pixels per dimension)
    DX_NM = 0.1             # Grid spacing (nm/pixel)
    DOMAIN_LENGTH_NM = 12.8 # Total physical domain size (nm)
    HALF_LENGTH_NM = 6.4    # Half-domain for centered coordinates
    
    @staticmethod
    def get_domain_extent():
        """Returns matplotlib extent for centered domain: [xmin, xmax, ymin, ymax]"""
        return [-SimulationDomain.HALF_LENGTH_NM, 
                SimulationDomain.HALF_LENGTH_NM,
                -SimulationDomain.HALF_LENGTH_NM,
                SimulationDomain.HALF_LENGTH_NM]
    
    @staticmethod
    def get_coordinate_arrays():
        """Returns physical coordinate arrays in nm"""
        x = np.linspace(-SimulationDomain.HALF_LENGTH_NM, 
                       SimulationDomain.HALF_LENGTH_NM,
                       SimulationDomain.N_POINTS)
        y = np.linspace(-SimulationDomain.HALF_LENGTH_NM,
                       SimulationDomain.HALF_LENGTH_NM,
                       SimulationDomain.N_POINTS)
        return x, y
    
    @staticmethod
    def get_domain_info():
        """Returns domain metadata for display/export"""
        return {
            'n_points': SimulationDomain.N_POINTS,
            'dx_nm': SimulationDomain.DX_NM,
            'domain_length_nm': SimulationDomain.DOMAIN_LENGTH_NM,
            'half_length_nm': SimulationDomain.HALF_LENGTH_NM,
            'area_nm2': SimulationDomain.DOMAIN_LENGTH_NM ** 2,
            'pixel_area_nm2': SimulationDomain.DX_NM ** 2,
            'total_pixels': SimulationDomain.N_POINTS ** 2
        }

# =============================================
# ENHANCED HEAT MAP VISUALIZER WITH PHYSICAL COORDINATES
# =============================================
class HeatMapVisualizer:
    """Enhanced heat map visualizer with diffusion visualization and physical coordinates"""
    def __init__(self):
        # --- EXTENDED COLORMAP LIST (unchanged) ---
        self.colormaps = { ... }  # (same as original)
        self.diffusion_physics = DiffusionPhysics()
        # Store domain parameters
        self.domain = SimulationDomain()
    
    def _apply_label(self, ax, default_text, override_text, func_name, **kwargs):
        """Applies text to axis or removes it if override is empty string."""
        if override_text is not None:
            if override_text.strip() == "":
                getattr(ax, func_name)(None)
            else:
                getattr(ax, func_name)(override_text, **kwargs)
        else:
            getattr(ax, func_name)(default_text, **kwargs)
    
    def create_stress_heatmap(self, stress_field, title="Stress Heat Map",
                            cmap_name='viridis', figsize=(12, 10),
                            colorbar_label="Stress (GPa)", vmin=None, vmax=None,
                            show_stats=True, target_angle=None, defect_type=None,
                            show_colorbar=True, aspect_ratio='equal', dpi=300,
                            label_config=None):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        if cmap_name not in plt.colormaps():
            cmap = plt.get_cmap('viridis')
        else:
            cmap = plt.get_cmap(cmap_name)
        
        if vmin is None:
            vmin = np.nanmin(stress_field)
        if vmax is None:
            vmax = np.nanmax(stress_field)
        
        # === KEY MODIFICATION: Use physical extent ===
        extent = self.domain.get_domain_extent()
        im = ax.imshow(stress_field, cmap=cmap, vmin=vmin, vmax=vmax,
                      aspect=aspect_ratio, interpolation='bilinear', 
                      origin='lower', extent=extent)  # <-- PHYSICAL COORDINATES
        
        # Customization Logic (unchanged)
        show_cb = show_colorbar
        if label_config:
            show_cb = not label_config.get('hide_colorbar', False)
        if show_cb:
            cbar_label_text = colorbar_label
            if label_config and 'colorbar_label' in label_config:
                cbar_label_text = label_config['colorbar_label']
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if cbar_label_text is not None and cbar_label_text != "":
                cbar.set_label(cbar_label_text, fontsize=16, fontweight='bold')
            cbar.ax.tick_params(labelsize=14)
        
        title_str = title
        if target_angle is not None and defect_type is not None:
            title_str = f"{title}\nθ = {target_angle:.1f}°, Defect: {defect_type}"
        if label_config and 'title_suffix' in label_config:
            if label_config['title_suffix']:
                title_str += f" {label_config['title_suffix']}"
        if label_config and 'hide_title' in label_config and label_config['hide_title']:
            ax.set_title(None)
        else:
            ax.set_title(title_str, fontsize=20, fontweight='bold', pad=20)
        
        # === UPDATED AXIS LABELS WITH PHYSICAL UNITS ===
        xlabel_text = label_config['xlabel'] if label_config and 'xlabel' in label_config else "X Position (nm)"
        ylabel_text = label_config['ylabel'] if label_config and 'ylabel' in label_config else "Y Position (nm)"
        self._apply_label(ax, "X Position (nm)", xlabel_text, 'set_xlabel', fontsize=16, fontweight='bold')
        self._apply_label(ax, "Y Position (nm)", ylabel_text, 'set_ylabel', fontsize=16, fontweight='bold')
        
        show_grid = True
        if label_config:
            show_grid = not label_config.get('hide_grid', False)
        if show_grid:
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray')
        else:
            ax.grid(False)
        
        if show_stats:
            stats_text = (f"Max: {vmax:.3f} GPa\nMin: {vmin:.3f} GPa\n"
                         f"Mean: {np.nanmean(stress_field):.3f} GPa\n"
                         f"Std: {np.nanstd(stress_field):.3f} GPa\n"
                         f"Domain: {self.domain.DOMAIN_LENGTH_NM} nm × {self.domain.DOMAIN_LENGTH_NM} nm")
            show_stats_box = True
            if label_config:
                show_stats_box = not label_config.get('hide_stats_box', False)
            if show_stats_box:
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       fontsize=12, fontweight='bold', verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()
        return fig
    
    def create_diffusion_heatmap(self, sigma_hydro_field, title="Diffusion Enhancement Map",
                               T_K=650, material='Silver', cmap_name='RdBu_r',
                               figsize=(12, 10), dpi=300, log_scale=True,
                               show_stats=True, target_angle=None, defect_type=None,
                               show_colorbar=True, aspect_ratio='equal',
                               model='physics_corrected', label_config=None):
        D_ratio = DiffusionPhysics.compute_diffusion_enhancement(
            sigma_hydro_field, T_K, material, model
        )
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        if cmap_name in plt.colormaps():
            cmap = plt.get_cmap(cmap_name)
        else:
            cmap = plt.get_cmap('RdBu_r')
        
        # === KEY MODIFICATION: Use physical extent ===
        extent = self.domain.get_domain_extent()
        
        if log_scale:
            log_data = np.log10(np.clip(D_ratio, 0.1, 10))
            vmin, vmax = -1, 1
            im = ax.imshow(log_data, cmap=cmap, vmin=vmin, vmax=vmax,
                          aspect=aspect_ratio, interpolation='bilinear', 
                          origin='lower', extent=extent)  # <-- PHYSICAL COORDINATES
        else:
            vmin, vmax = 0.1, 10
            im = ax.imshow(D_ratio, cmap=cmap, vmin=vmin, vmax=vmax,
                          aspect=aspect_ratio, interpolation='bilinear', 
                          origin='lower', extent=extent,  # <-- PHYSICAL COORDINATES
                          norm=LogNorm(vmin=vmin, vmax=vmax) if log_scale else None)
        
        # (Rest of function remains similar with physical units in labels)
        # ... [colorbar, title, axis labels with "(nm)" units] ...
        
        if show_stats:
            enhancement_regions = D_ratio > 1.0
            suppression_regions = D_ratio < 1.0
            stats_text = (f"Max Enhancement: {np.max(D_ratio):.2f}x\n"
                         f"Min (Suppression): {np.min(D_ratio):.2f}x\n"
                         f"Mean: {np.mean(D_ratio):.2f}x\n"
                         f"Enhanced Area: {np.sum(enhancement_regions)/D_ratio.size*100:.1f}%\n"
                         f"Suppressed Area: {np.sum(suppression_regions)/D_ratio.size*100:.1f}%\n"
                         f"Domain: {self.domain.DOMAIN_LENGTH_NM} nm × {self.domain.DOMAIN_LENGTH_NM} nm")
            # ... [rest of stats display] ...
        
        plt.tight_layout()
        return fig, D_ratio
    
    def create_comparison_dashboard(self, interpolated_fields, source_fields, source_info,
                                  target_angle, defect_type, component='von_mises',
                                  cmap_name='viridis', figsize=(24, 18),
                                  ground_truth_index=None, defect_type_filter=None,
                                  label_config=None):
        """Create comprehensive comparison dashboard with PHYSICAL COORDINATES"""
        # ... [filtering logic unchanged] ...
        
        fig = plt.figure(figsize=figsize, dpi=300)
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4, left=0.05, right=0.95, top=0.92, bottom=0.05)
        
        # Get physical extent for all plots
        extent = self.domain.get_domain_extent()
        
        # HELPER FOR SUBPLOT CUSTOMIZATION WITH PHYSICAL UNITS
        def customize_ax(ax, title, xlabel="X Position (nm)", ylabel="Y Position (nm)"):
            if label_config:
                if label_config.get('hide_title', False): 
                    ax.set_title(None)
                elif 'title_suffix' in label_config and label_config['title_suffix']:
                    ax.set_title(f"{title} {label_config['title_suffix']}", 
                               fontsize=14, fontweight='bold', pad=10)
                else: 
                    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
            else: 
                ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
            
            x_lbl = label_config['xlabel'] if label_config and 'xlabel' in label_config else xlabel
            y_lbl = label_config['ylabel'] if label_config and 'ylabel' in label_config else ylabel
            self._apply_label(ax, xlabel, x_lbl, 'set_xlabel', fontsize=11)
            self._apply_label(ax, ylabel, y_lbl, 'set_ylabel', fontsize=11)
            
            if label_config and label_config.get('hide_grid', False): 
                ax.grid(False)
            else: 
                ax.grid(True, alpha=0.2)
        
        # PLOT 1: Interpolated (with physical extent)
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(interpolated_fields[component], cmap=cmap_name,
                        vmin=vmin, vmax=vmax, aspect='equal', 
                        interpolation='bilinear', origin='lower', extent=extent)  # <-- PHYSICAL COORDINATES
        # ... [rest of plot customization with "(nm)" labels] ...
        
        # PLOT 2: Ground Truth (with physical extent)
        ax2 = fig.add_subplot(gs[0, 1])
        if ground_truth_index is not None and 0 <= ground_truth_index < len(source_fields):
            gt_field = source_fields[ground_truth_index].get(component)
            if gt_field is not None:
                gt_theta = source_info['theta_degrees'][ground_truth_index]
                gt_distance = source_info['distances'][ground_truth_index]
                im2 = ax2.imshow(gt_field, cmap=cmap_name,
                                vmin=vmin, vmax=vmax, aspect='equal', 
                                interpolation='bilinear', origin='lower', extent=extent)  # <-- PHYSICAL COORDINATES
                # ... [rest unchanged] ...
        
        # PLOT 3: Difference (with physical extent)
        ax3 = fig.add_subplot(gs[0, 2])
        if ground_truth_index is not None and 0 <= ground_truth_index < len(source_fields):
            gt_field = source_fields[ground_truth_index].get(component)
            if gt_field is not None:
                diff_field = interpolated_fields[component] - gt_field
                max_diff = np.max(np.abs(diff_field))
                im3 = ax3.imshow(diff_field, cmap='RdBu_r',
                                vmin=-max_diff, vmax=max_diff, aspect='equal',
                                interpolation='bilinear', origin='lower', extent=extent)  # <-- PHYSICAL COORDINATES
                # ... [rest unchanged] ...
        
        # PLOTS 4-8: Statistical plots remain unchanged (no spatial coordinates)
        # ... [rest of function] ...
        
        # Global Suptitle with domain info
        if not (label_config and label_config.get('hide_suptitle', False)):
            suptitle_txt = (f"Theory-Informed Interpolation: Target θ={target_angle:.1f}°, {defect_type} | "
                          f"Domain: {self.domain.DOMAIN_LENGTH_NM} nm × {self.domain.DOMAIN_LENGTH_NM} nm")
            if label_config and 'suptitle' in label_config and label_config['suptitle']:
                suptitle_txt = label_config['suptitle']
            plt.suptitle(suptitle_txt, fontsize=22, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        return fig
    
    # Other methods similarly updated with extent parameter in imshow calls
    # ...
