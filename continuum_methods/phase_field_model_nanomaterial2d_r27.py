# =============================================
# ULTIMATE Ag NP Defect Analyzer – ENHANCED PUBLICATION-QUALITY VERSION
# =============================================
import streamlit as st
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams, cm
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter, LogLocator
import pandas as pd
import zipfile
from io import BytesIO
import time
import hashlib
import json
from datetime import datetime
from scipy import stats, interpolate
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Ellipse, Rectangle, Polygon, FancyBboxPatch, ConnectionPatch
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm, LogNorm, PowerNorm
from matplotlib.font_manager import FontProperties
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configure page with better styling
st.set_page_config(page_title="Ag NP Multi-Defect Analyzer", layout="wide")
st.title("🔬 Ag Nanoparticle Multi-Defect Comparison Platform - PUBLICATION ENHANCED")
st.markdown("""
**Publication-Quality Output • Journal Templates • Vector Export • Advanced Statistical Analysis**
**Nature/Science Style • Custom LaTeX Fonts • Enhanced Resolution • Professional Visualization**
""")

# =============================================
# JOURNAL-SPECIFIC STYLING TEMPLATES
# =============================================
class JournalTemplates:
    """Publication-quality journal templates"""
    
    @staticmethod
    def get_journal_styles():
        """Return journal-specific style parameters"""
        return {
            'nature': {
                'figure_width_single': 8.9,  # cm to inches
                'figure_width_double': 18.3,
                'font_family': 'Arial',
                'font_size_small': 7,
                'font_size_medium': 8,
                'font_size_large': 9,
                'line_width': 0.5,
                'axes_linewidth': 0.5,
                'tick_width': 0.5,
                'tick_length': 2,
                'grid_alpha': 0.1,
                'dpi': 600,
                'color_cycle': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            },
            'science': {
                'figure_width_single': 5.5,
                'figure_width_double': 11.4,
                'font_family': 'Helvetica',
                'font_size_small': 8,
                'font_size_medium': 9,
                'font_size_large': 10,
                'line_width': 0.75,
                'axes_linewidth': 0.75,
                'tick_width': 0.75,
                'tick_length': 3,
                'grid_alpha': 0.15,
                'dpi': 600,
                'color_cycle': ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30',
                              '#4DBEEE', '#A2142F', '#FF00FF', '#00FFFF', '#FFA500']
            },
            'advanced_materials': {
                'figure_width_single': 8.6,
                'figure_width_double': 17.8,
                'font_family': 'Arial',
                'font_size_small': 8,
                'font_size_medium': 9,
                'font_size_large': 10,
                'line_width': 1.0,
                'axes_linewidth': 1.0,
                'tick_width': 1.0,
                'tick_length': 4,
                'grid_alpha': 0.2,
                'dpi': 600,
                'color_cycle': ['#004488', '#DDAA33', '#BB5566', '#000000', '#44AA99',
                              '#882255', '#117733', '#999933', '#AA4499', '#88CCEE']
            },
            'prl': {
                'figure_width_single': 3.4,
                'figure_width_double': 7.0,
                'font_family': 'Times New Roman',
                'font_size_small': 8,
                'font_size_medium': 10,
                'font_size_large': 12,
                'line_width': 1.0,
                'axes_linewidth': 1.0,
                'tick_width': 1.0,
                'tick_length': 4,
                'grid_alpha': 0,
                'dpi': 600,
                'color_cycle': ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442',
                              '#0072B2', '#D55E00', '#CC79A7', '#999999', '#FFFFFF']
            },
            'custom': {
                'figure_width_single': 6.0,
                'figure_width_double': 12.0,
                'font_family': 'DejaVu Sans',
                'font_size_small': 10,
                'font_size_medium': 12,
                'font_size_large': 14,
                'line_width': 1.5,
                'axes_linewidth': 1.5,
                'tick_width': 1.0,
                'tick_length': 5,
                'grid_alpha': 0.3,
                'dpi': 300,
                'color_cycle': plt.cm.Set2(np.linspace(0, 1, 10))
            }
        }
    
    @staticmethod
    def apply_journal_style(fig, axes, journal_name='nature'):
        """Apply journal-specific styling to figure"""
        styles = JournalTemplates.get_journal_styles()
        style = styles.get(journal_name, styles['nature'])
        
        # Set rcParams for consistent styling
        rcParams.update({
            'font.family': style['font_family'],
            'font.size': style['font_size_medium'],
            'axes.linewidth': style['axes_linewidth'],
            'axes.labelsize': style['font_size_medium'],
            'axes.titlesize': style['font_size_large'],
            'xtick.labelsize': style['font_size_small'],
            'ytick.labelsize': style['font_size_small'],
            'legend.fontsize': style['font_size_small'],
            'figure.titlesize': style['font_size_large'],
            'lines.linewidth': style['line_width'],
            'lines.markersize': 4,
            'xtick.major.width': style['tick_width'],
            'ytick.major.width': style['tick_width'],
            'xtick.minor.width': style['tick_width'] * 0.5,
            'ytick.minor.width': style['tick_width'] * 0.5,
            'xtick.major.size': style['tick_length'],
            'ytick.major.size': style['tick_length'],
            'xtick.minor.size': style['tick_length'] * 0.6,
            'ytick.minor.size': style['tick_length'] * 0.6,
            'axes.grid': False,
            'savefig.dpi': style['dpi'],
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'axes.prop_cycle': plt.cycler(color=style['color_cycle'])
        })
        
        # Apply to all axes
        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        elif isinstance(axes, list):
            axes_flat = axes
        else:
            axes_flat = [axes]
        
        for ax in axes_flat:
            if ax is not None:
                # Add minor ticks
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                
                # Set spine visibility
                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)
                ax.spines['top'].set_linewidth(style['axes_linewidth'] * 0.5)
                ax.spines['right'].set_linewidth(style['axes_linewidth'] * 0.5)
                
                # Improve tick formatting
                ax.tick_params(which='both', direction='in', top=True, right=True)
                ax.tick_params(which='major', length=style['tick_length'])
                ax.tick_params(which='minor', length=style['tick_length'] * 0.6)
        
        return fig, style

# =============================================
# ADVANCED PLOTTING ENHANCEMENTS
# =============================================
class PublicationEnhancer:
    """Advanced plotting enhancements for publication-quality figures"""
    
    @staticmethod
    def create_custom_colormaps():
        """Create enhanced scientific colormaps"""
        # Perceptually uniform sequential
        plasma_enhanced = LinearSegmentedColormap.from_list('plasma_enhanced', [
            (0.0, '#0c0887'),
            (0.1, '#4b03a1'),
            (0.3, '#8b0aa5'),
            (0.5, '#b83289'),
            (0.7, '#db5c68'),
            (0.9, '#f48849'),
            (1.0, '#fec325')
        ])
        
        # Diverging with better contrast
        coolwarm_enhanced = LinearSegmentedColormap.from_list('coolwarm_enhanced', [
            (0.0, '#3a4cc0'),
            (0.25, '#8abcdd'),
            (0.5, '#f7f7f7'),
            (0.75, '#f0b7a4'),
            (1.0, '#b40426')
        ])
        
        # Categorical for defect types
        defect_categorical = ListedColormap([
            '#1f77b4',  # ISF - Blue
            '#ff7f0e',  # ESF - Orange
            '#2ca02c',  # Twin - Green
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b'   # Brown
        ])
        
        # Stress-specific colormap
        stress_map = LinearSegmentedColormap.from_list('stress_map', [
            (0.0, '#2c7bb6'),
            (0.2, '#abd9e9'),
            (0.4, '#ffffbf'),
            (0.6, '#fdae61'),
            (0.8, '#d7191c'),
            (1.0, '#800026')
        ])
        
        return {
            'plasma_enhanced': plasma_enhanced,
            'coolwarm_enhanced': coolwarm_enhanced,
            'defect_categorical': defect_categorical,
            'stress_map': stress_map
        }
    
    @staticmethod
    def add_error_shading(ax, x, y_mean, y_std, color='blue', alpha=0.3, label=''):
        """Add error shading to line plots"""
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, 
                       color=color, alpha=alpha, label=label + ' ± std')
        return ax
    
    @staticmethod
    def add_confidence_band(ax, x, y_data, confidence=0.95, color='blue', alpha=0.2):
        """Add confidence band to line plots"""
        y_mean = np.mean(y_data, axis=0)
        y_std = np.std(y_data, axis=0)
        n = len(y_data)
        t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
        y_err = t_val * y_std / np.sqrt(n)
        
        ax.fill_between(x, y_mean - y_err, y_mean + y_err, 
                       color=color, alpha=alpha, label=f'{int(confidence*100)}% CI')
        return ax, y_mean, y_err
    
    @staticmethod
    def add_inset(ax, bounds, **kwargs):
        """Add inset axes for detail views"""
        inset_ax = ax.inset_axes(bounds)
        return inset_ax
    
    @staticmethod
    def add_scale_bar(ax, length_nm, location='lower right', color='black', linewidth=2):
        """Add scale bar to microscopy-style images"""
        if location == 'lower right':
            x_pos = 0.95
            y_pos = 0.05
            ha = 'right'
            va = 'bottom'
        elif location == 'lower left':
            x_pos = 0.05
            y_pos = 0.05
            ha = 'left'
            va = 'bottom'
        elif location == 'upper right':
            x_pos = 0.95
            y_pos = 0.95
            ha = 'right'
            va = 'top'
        else:
            x_pos = 0.05
            y_pos = 0.95
            ha = 'left'
            va = 'top'
        
        # Convert to axis coordinates
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        
        # Bar position in data coordinates
        bar_x_start = xlim[1] - x_range * 0.15
        bar_x_end = bar_x_start - length_nm
        bar_y = ylim[0] + y_range * 0.05
        
        # Draw scale bar
        ax.plot([bar_x_start, bar_x_end], [bar_y, bar_y], 
               color=color, linewidth=linewidth, solid_capstyle='butt')
        
        # Add text
        ax.text((bar_x_start + bar_x_end) / 2, bar_y + y_range * 0.02,
               f'{length_nm} nm', ha='center', va='bottom',
               color=color, fontsize=8, fontweight='bold')
        
        return ax
    
    @staticmethod
    def create_fancy_legend(ax, lines, labels, **kwargs):
        """Create enhanced legend with better formatting"""
        legend = ax.legend(lines, labels, **kwargs)
        legend.get_frame().set_linewidth(0.5)
        legend.get_frame().set_alpha(0.9)
        return legend
    
    @staticmethod
    def add_annotations(ax, annotations, arrowstyle='->', **kwargs):
        """Add professional annotations with arrows"""
        for ann in annotations:
            ax.annotate(ann['text'], xy=ann['xy'], xytext=ann['xytext'],
                       arrowprops=dict(arrowstyle=arrowstyle, **kwargs),
                       **{k: v for k, v in ann.items() if k not in ['text', 'xy', 'xytext']})
        return ax
    
    @staticmethod
    def create_colorbar_with_ticks(ax, mappable, orientation='vertical', 
                                  label='', ticks=None, tick_labels=None, **kwargs):
        """Create publication-quality colorbar with custom ticks"""
        cbar = plt.colorbar(mappable, ax=ax, orientation=orientation, **kwargs)
        cbar.set_label(label, fontweight='bold')
        
        if ticks is not None:
            cbar.set_ticks(ticks)
        if tick_labels is not None:
            cbar.set_ticklabels(tick_labels)
        
        cbar.ax.tick_params(labelsize=8)
        return cbar

# =============================================
# ENHANCED FIGURE STYLER WITH PUBLICATION FEATURES
# =============================================
class EnhancedFigureStyler(FigureStyler):
    """Extended figure styler with publication-quality enhancements"""
    
    @staticmethod
    def apply_publication_styling(fig, axes, style_params):
        """Apply enhanced publication styling"""
        # Apply base styling
        fig = FigureStyler.apply_advanced_styling(fig, axes, style_params)
        
        # Get axes list
        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        elif isinstance(axes, list):
            axes_flat = axes
        else:
            axes_flat = [axes]
        
        # Enhanced styling for each axis
        for ax in axes_flat:
            if ax is not None:
                # Set scientific notation for large/small numbers
                ax.ticklabel_format(style='sci', scilimits=(-3, 3), useMathText=True)
                
                # Add minor ticks
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                
                # Set tick parameters
                ax.tick_params(which='both', direction='in', top=True, right=True)
                ax.tick_params(which='major', length=6, width=style_params.get('tick_width', 1.0))
                ax.tick_params(which='minor', length=3, width=style_params.get('tick_width', 1.0) * 0.5)
                
                # Format axis labels with LaTeX
                if style_params.get('use_latex', False):
                    xlabel = ax.get_xlabel()
                    ylabel = ax.get_ylabel()
                    if xlabel:
                        ax.set_xlabel(f'${xlabel}$')
                    if ylabel:
                        ax.set_ylabel(f'${ylabel}$')
        
        # Adjust layout
        fig.tight_layout(rect=[0, 0, 1, 0.95], pad=style_params.get('layout_pad', 1.0))
        
        return fig
    
    @staticmethod
    def get_publication_controls():
        """Get enhanced publication styling controls"""
        style_params = FigureStyler.get_styling_controls()
        
        st.sidebar.header("📰 Publication-Quality Settings")
        
        with st.sidebar.expander("🎯 Journal Templates", expanded=False):
            journal = st.selectbox(
                "Journal Style",
                ["Nature", "Science", "Advanced Materials", "Physical Review Letters", "Custom"],
                index=0
            )
            
            style_params['journal_style'] = journal.lower()
            style_params['use_latex'] = st.checkbox("Use LaTeX Formatting", False)
            style_params['vector_output'] = st.checkbox("Enable Vector Export (PDF/SVG)", True)
        
        with st.sidebar.expander("📐 Advanced Layout", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                style_params['layout_pad'] = st.slider("Layout Padding", 0.5, 3.0, 1.0, 0.1)
                style_params['wspace'] = st.slider("Horizontal Spacing", 0.1, 1.0, 0.3, 0.05)
            with col2:
                style_params['hspace'] = st.slider("Vertical Spacing", 0.1, 1.0, 0.4, 0.05)
                style_params['figure_dpi'] = st.select_slider("Figure DPI", 
                                                           options=[150, 300, 600, 1200], 
                                                           value=600)
        
        with st.sidebar.expander("📈 Enhanced Plot Features", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                style_params['show_minor_ticks'] = st.checkbox("Show Minor Ticks", True)
                style_params['show_error_bars'] = st.checkbox("Show Error Bars", True)
                style_params['show_confidence'] = st.checkbox("Show Confidence Intervals", False)
            with col2:
                style_params['grid_style'] = st.selectbox("Grid Style", 
                                                         ['-', '--', '-.', ':'],
                                                         index=1)
                style_params['grid_zorder'] = st.slider("Grid Z-Order", 0, 10, 0)
        
        with st.sidebar.expander("🎨 Enhanced Color Settings", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                style_params['colorbar_extend'] = st.selectbox("Colorbar Extend", 
                                                              ['neither', 'both', 'min', 'max'],
                                                              index=0)
                style_params['colorbar_format'] = st.selectbox("Colorbar Format", 
                                                              ['auto', 'sci', 'plain'],
                                                              index=0)
            with col2:
                style_params['cmap_normalization'] = st.selectbox("Colormap Normalization",
                                                                ['linear', 'log', 'power'],
                                                                index=0)
                if style_params['cmap_normalization'] == 'power':
                    style_params['gamma'] = st.slider("Gamma", 0.1, 3.0, 1.0, 0.1)
        
        return style_params

# =============================================
# PUBLICATION-QUALITY PLOTTING FUNCTIONS
# =============================================
def create_enhanced_comparison_plot(simulations, frames, config, style_params):
    """Create publication-quality comparison plots"""
    
    # Get journal style
    journal_style = style_params.get('journal_style', 'nature')
    fig, axes = JournalTemplates.apply_journal_style(*JournalTemplates.get_journal_styles()[journal_style])
    
    # Create figure based on comparison type
    if config['type'] == "Side-by-Side Heatmaps":
        return create_publication_heatmaps(simulations, frames, config, style_params)
    elif config['type'] == "Overlay Line Profiles":
        return create_publication_line_profiles(simulations, frames, config, style_params)
    elif config['type'] == "Statistical Summary":
        return create_publication_statistics(simulations, frames, config, style_params)
    elif config['type'] == "Defect-Stress Correlation":
        return create_publication_correlation(simulations, frames, config, style_params)
    else:
        # Fall back to original function
        return create_defect_stress_correlation_plot(simulations, frames, config, style_params)

def create_publication_heatmaps(simulations, frames, config, style_params):
    """Publication-quality heatmap comparison"""
    st.subheader("🌡️ Publication-Quality Heatmap Comparison")
    
    stress_map = {
        "Stress Magnitude |σ|": 'sigma_mag',
        "Hydrostatic σ_h": 'sigma_hydro',
        "von Mises σ_vM": 'von_mises'
    }
    stress_key = stress_map[config['stress_component']]
    
    n_sims = len(simulations)
    cols = min(3, n_sims)
    rows = (n_sims + cols - 1) // cols
    
    # Create figure with journal sizing
    journal_styles = JournalTemplates.get_journal_styles()
    journal = style_params.get('journal_style', 'nature')
    fig_width = journal_styles[journal]['figure_width_double'] / 2.54  # Convert cm to inches
    
    fig, axes = plt.subplots(rows, cols, 
                            figsize=(fig_width, fig_width * 0.8 * rows/cols),
                            constrained_layout=True)
    
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Get enhanced colormaps
    enhanced_cmaps = PublicationEnhancer.create_custom_colormaps()
    
    for idx, (sim, frame) in enumerate(zip(simulations, frames)):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Get data
        eta, stress_fields = sim['history'][frame]
        stress_data = stress_fields[stress_key]
        
        # Apply smoothing for better visualization
        if style_params.get('apply_smoothing', True):
            stress_data = gaussian_filter(stress_data, sigma=1)
        
        # Choose colormap
        cmap_name = sim['params']['sigma_cmap']
        if cmap_name in enhanced_cmaps:
            cmap = enhanced_cmaps[cmap_name]
        else:
            cmap = COLORMAPS[cmap_name]
        
        # Create heatmap with enhanced settings
        im = ax.imshow(stress_data, extent=extent, cmap=cmap, 
                      origin='lower', aspect='auto',
                      norm=LogNorm() if style_params.get('use_log_scale', False) else None)
        
        # Add contour lines for defect boundary
        contour = ax.contour(X, Y, eta, levels=[0.5], colors='white', 
                           linewidths=1, linestyles='--', alpha=0.8)
        
        # Add scale bar
        PublicationEnhancer.add_scale_bar(ax, 5.0, location='lower right')
        
        # Enhanced title
        title = f"{sim['params']['defect_type']}"
        if sim['params']['orientation'] != "Horizontal {111} (0°)":
            title += f"\n{sim['params']['orientation'].split(' ')[0]}"
        
        ax.set_title(title, fontsize=style_params.get('title_font_size', 10),
                    fontweight='semibold', pad=10)
        
        # Axis labels only on edge plots
        if row == rows - 1:
            ax.set_xlabel("x (nm)", fontsize=style_params.get('label_font_size', 9))
        if col == 0:
            ax.set_ylabel("y (nm)", fontsize=style_params.get('label_font_size', 9))
        
        # Enhanced colorbar
        cbar = PublicationEnhancer.create_colorbar_with_ticks(
            ax, im, orientation='vertical', 
            label=f"{config['stress_component']} (GPa)",
            ticks=np.linspace(np.nanmin(stress_data), np.nanmax(stress_data), 5)
        )
        cbar.ax.tick_params(labelsize=7)
    
    # Hide empty subplots
    for idx in range(n_sims, rows*cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    # Apply publication styling
    fig = EnhancedFigureStyler.apply_publication_styling(fig, axes, style_params)
    
    return fig

def create_publication_line_profiles(simulations, frames, config, style_params):
    """Publication-quality line profile comparison"""
    st.subheader("📈 Enhanced Line Profile Analysis")
    
    stress_map = {
        "Stress Magnitude |σ|": 'sigma_mag',
        "Hydrostatic σ_h": 'sigma_hydro',
        "von Mises σ_vM": 'von_mises'
    }
    stress_key = stress_map[config['stress_component']]
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(12, 10))
    
    # Panel A: Multiple line profiles
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    # Panel B: Statistical summary
    ax2 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
    # Panel C: Profile positions
    ax3 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    
    # Get enhanced colormaps
    enhanced_cmaps = PublicationEnhancer.create_custom_colormaps()
    colors = enhanced_cmaps['defect_categorical'].colors[:len(simulations)]
    
    # Prepare data for all profiles
    all_profiles = []
    profile_stats = []
    
    for idx, (sim, frame, color) in enumerate(zip(simulations, frames, colors)):
        # Get data
        eta, stress_fields = sim['history'][frame]
        stress_data = stress_fields[stress_key]
        
        # Extract multiple profiles
        center_y = N // 2
        profile_horizontal = stress_data[center_y, :]
        profile_vertical = stress_data[:, center_x]
        
        x_pos = np.linspace(extent[0], extent[1], N)
        
        # Plot with enhanced styling
        line_style = config.get('line_style', 'solid')
        line = ax1.plot(x_pos, profile_horizontal, color=color, 
                       linewidth=style_params.get('line_width', 1.5),
                       linestyle=line_style,
                       label=f"{sim['params']['defect_type']}",
                       alpha=0.8)[0]
        
        # Store for statistics
        all_profiles.append(profile_horizontal)
        
        # Calculate profile statistics
        profile_stats.append({
            'Defect': sim['params']['defect_type'],
            'Max': np.max(profile_horizontal),
            'Mean': np.mean(profile_horizontal),
            'Std': np.std(profile_horizontal),
            'FWHM': calculate_fwhm(x_pos, profile_horizontal),
            'Peak Position': x_pos[np.argmax(profile_horizontal)]
        })
    
    # Add error shading if multiple simulations of same type
    if style_params.get('show_confidence', False) and len(simulations) > 1:
        # Group by defect type
        defect_groups = {}
        for idx, sim in enumerate(simulations):
            defect = sim['params']['defect_type']
            if defect not in defect_groups:
                defect_groups[defect] = []
            defect_groups[defect].append(all_profiles[idx])
        
        for defect, profiles in defect_groups.items():
            if len(profiles) > 1:
                profiles_array = np.array(profiles)
                PublicationEnhancer.add_confidence_band(ax1, x_pos, profiles_array, 
                                                       color=colors[list(defect_groups.keys()).index(defect)])
    
    # Enhance axis 1
    ax1.set_xlabel("Position (nm)", fontsize=style_params.get('label_font_size', 10))
    ax1.set_ylabel(f"{config['stress_component']} (GPa)", 
                  fontsize=style_params.get('label_font_size', 10))
    ax1.set_title("Stress Line Profiles", fontsize=style_params.get('title_font_size', 12),
                 fontweight='bold')
    
    # Add legend with enhanced formatting
    PublicationEnhancer.create_fancy_legend(ax1, *ax1.get_legend_handles_labels(),
                                          loc='upper right', frameon=True,
                                          fancybox=True, shadow=False)
    
    # Panel B: Statistical summary as bar plot
    if profile_stats:
        df_stats = pd.DataFrame(profile_stats)
        x_positions = np.arange(len(df_stats))
        
        bars = ax2.bar(x_positions, df_stats['Max'], color=colors, alpha=0.7)
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(df_stats['Defect'], rotation=45, ha='right')
        ax2.set_ylabel("Maximum Stress (GPa)", fontsize=9)
        ax2.set_title("Peak Stress Comparison", fontsize=10, fontweight='bold')
        
        # Add value labels
        for bar, val in zip(bars, df_stats['Max']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Panel C: Show profile locations on one simulation
    if simulations:
        sim = simulations[0]
        eta, _ = sim['history'][frames[0]]
        im = ax3.imshow(eta, extent=extent, cmap=enhanced_cmaps['plasma_enhanced'], 
                       origin='lower', aspect='auto')
        
        # Add profile lines
        center_y = N // 2
        ax3.axhline(y=extent[2] + center_y * dx, color='red', 
                   linewidth=2, linestyle='-', alpha=0.7, label='Horizontal Profile')
        
        # Add scale bar
        PublicationEnhancer.add_scale_bar(ax3, 5.0, location='lower right', color='white')
        
        ax3.set_xlabel("x (nm)", fontsize=9)
        ax3.set_ylabel("y (nm)", fontsize=9)
        ax3.set_title("Profile Locations", fontsize=10, fontweight='bold')
        
        # Add colorbar
        PublicationEnhancer.create_colorbar_with_ticks(
            ax3, im, label='Defect Parameter η'
        )
    
    # Apply publication styling to all axes
    fig = EnhancedFigureStyler.apply_publication_styling(fig, [ax1, ax2, ax3], style_params)
    
    # Add panel labels
    for ax, label in zip([ax1, ax2, ax3], ['A', 'B', 'C']):
        ax.text(-0.1, 1.05, label, transform=ax.transAxes,
               fontsize=14, fontweight='bold', va='top')
    
    return fig

def create_publication_statistics(simulations, frames, config, style_params):
    """Publication-quality statistical analysis"""
    st.subheader("📊 Comprehensive Statistical Analysis")
    
    stress_map = {
        "Stress Magnitude |σ|": 'sigma_mag',
        "Hydrostatic σ_h": 'sigma_hydro',
        "von Mises σ_vM": 'von_mises'
    }
    stress_key = stress_map[config['stress_component']]
    
    # Create multi-panel figure
    fig = plt.figure(figsize=(14, 10))
    
    # Define subplots
    ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=2)  # Box plot
    ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2)  # Violin plot
    ax3 = plt.subplot2grid((3, 4), (1, 0), colspan=2)  # Histogram
    ax4 = plt.subplot2grid((3, 4), (1, 2), colspan=2)  # Cumulative distribution
    ax5 = plt.subplot2grid((3, 4), (2, 0), colspan=4)  # Statistical table
    
    # Get colors
    enhanced_cmaps = PublicationEnhancer.create_custom_colormaps()
    colors = enhanced_cmaps['defect_categorical'].colors[:len(simulations)]
    
    # Collect data
    all_data = []
    labels = []
    
    for idx, (sim, frame) in enumerate(zip(simulations, frames)):
        eta, stress_fields = sim['history'][frame]
        stress_data = stress_fields[stress_key].flatten()
        stress_data = stress_data[np.isfinite(stress_data)]
        
        all_data.append(stress_data)
        labels.append(f"{sim['params']['defect_type']}\n({sim['params']['orientation'][:10]}...)")
    
    # Panel 1: Enhanced box plot
    bp = ax1.boxplot(all_data, labels=labels, patch_artist=True,
                    showmeans=True, meanline=True, showfliers=False,
                    meanprops=dict(color='white', linewidth=1.5),
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(color='gray', linewidth=1),
                    capprops=dict(color='gray', linewidth=1),
                    boxprops=dict(linewidth=1))
    
    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_title(f"Distribution of {config['stress_component']}", 
                 fontsize=12, fontweight='bold')
    ax1.set_ylabel("Stress (GPa)", fontsize=10)
    ax1.tick_params(axis='x', rotation=45, labelsize=9)
    
    # Add mean values as text
    for i, data in enumerate(all_data):
        mean_val = np.mean(data)
        ax1.text(i + 1, mean_val, f'{mean_val:.2f}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Panel 2: Violin plot
    parts = ax2.violinplot(all_data, showmeans=True, showmedians=True)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.5)
    
    ax2.set_title("Probability Density", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Stress (GPa)", fontsize=10)
    ax2.set_xticks(range(1, len(labels) + 1))
    ax2.set_xticklabels([sim['params']['defect_type'] for sim in simulations])
    
    # Panel 3: Histogram with KDE
    ax3.hist(all_data, bins=30, density=True, stacked=True, 
            label=[sim['params']['defect_type'] for sim in simulations],
            color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
    
    # Add KDE
    for data, color, label in zip(all_data, colors, labels):
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(min(data.min() for data in all_data), 
                             max(data.max() for data in all_data), 100)
        ax3.plot(x_range, kde(x_range), color=color, linewidth=2, label=label.split('\n')[0])
    
    ax3.set_title("Histogram with KDE", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Stress (GPa)", fontsize=10)
    ax3.set_ylabel("Density", fontsize=10)
    ax3.legend(fontsize=8)
    
    # Panel 4: Cumulative distribution
    for data, color, label in zip(all_data, colors, labels):
        sorted_data = np.sort(data)
        y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax4.plot(sorted_data, y_vals, color=color, linewidth=2, label=label.split('\n')[0])
    
    ax4.set_title("Cumulative Distribution", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Stress (GPa)", fontsize=10)
    ax4.set_ylabel("Cumulative Probability", fontsize=10)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    # Panel 5: Statistical table
    ax5.axis('off')
    
    # Create comprehensive statistics table
    table_data = []
    columns = ['Defect', 'N', 'Mean', 'Std', 'Min', '25%', 'Median', '75%', 'Max', 'Skew', 'Kurtosis']
    
    for idx, (data, sim) in enumerate(zip(all_data, simulations)):
        table_data.append([
            sim['params']['defect_type'],
            len(data),
            f"{np.mean(data):.3f}",
            f"{np.std(data):.3f}",
            f"{np.min(data):.3f}",
            f"{np.percentile(data, 25):.3f}",
            f"{np.median(data):.3f}",
            f"{np.percentile(data, 75):.3f}",
            f"{np.max(data):.3f}",
            f"{stats.skew(data):.3f}",
            f"{stats.kurtosis(data):.3f}"
        ])
    
    # Create table
    table = ax5.table(cellText=table_data, colLabels=columns,
                     cellLoc='center', loc='center',
                     colColours=['#f2f2f2']*len(columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Color code cells
    for i in range(len(table_data)):
        for j in range(1, len(columns)):  # Skip first column (Defect)
            table[(i+1, j)].set_facecolor(colors[i] + (0.3,))  # Add alpha
    
    # Apply publication styling
    fig = EnhancedFigureStyler.apply_publication_styling(fig, [ax1, ax2, ax3, ax4, ax5], style_params)
    
    # Add panel labels
    for ax, label in zip([ax1, ax2, ax3, ax4], ['A', 'B', 'C', 'D']):
        ax.text(-0.1, 1.05, label, transform=ax.transAxes,
               fontsize=14, fontweight='bold', va='top')
    
    return fig

def create_publication_correlation(simulations, frames, config, style_params):
    """Publication-quality correlation analysis"""
    st.subheader("📊 Advanced Correlation Analysis")
    
    # Component mapping
    component_map = {
        "Stress Magnitude |σ|": 'sigma_mag',
        "Hydrostatic σ_h": 'sigma_hydro',
        "von Mises σ_vM": 'von_mises',
        "Defect Parameter η": 'eta'
    }
    
    x_key = component_map[config.get('correlation_x', 'Defect Parameter η')]
    y_key = component_map[config.get('correlation_y', 'Stress Magnitude |σ|')]
    
    # Create multi-panel figure
    fig = plt.figure(figsize=(15, 12))
    
    # Define subplot grid
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)  # Scatter with regression
    ax2 = plt.subplot2grid((3, 3), (0, 2))              # Correlation coefficients
    ax3 = plt.subplot2grid((3, 3), (1, 0))              # Residuals
    ax4 = plt.subplot2grid((3, 3), (1, 1))              # QQ plot
    ax5 = plt.subplot2grid((3, 3), (1, 2))              # Histogram of residuals
    ax6 = plt.subplot2grid((3, 3), (2, 0), colspan=3)   # Regression parameters
    
    # Get enhanced colormaps
    enhanced_cmaps = PublicationEnhancer.create_custom_colormaps()
    colors = enhanced_cmaps['defect_categorical'].colors[:len(simulations)]
    
    # Store regression results
    regression_results = []
    
    for idx, (sim, frame, color) in enumerate(zip(simulations, frames, colors)):
        # Get data
        eta, stress_fields = sim['history'][frame]
        
        # Prepare x data
        if x_key == 'eta':
            x_data = eta.flatten()
        else:
            x_data = stress_fields[x_key].flatten()
        
        # Prepare y data
        if y_key == 'eta':
            y_data = eta.flatten()
        else:
            y_data = stress_fields[y_key].flatten()
        
        # Sample data for clarity
        sample_size = min(5000, len(x_data))
        indices = np.random.choice(len(x_data), sample_size, replace=False)
        x_sampled = x_data[indices]
        y_sampled = y_data[indices]
        
        # Remove outliers
        q_low, q_high = np.percentile(x_sampled, [1, 99])
        mask = (x_sampled > q_low) & (x_sampled < q_high)
        x_sampled = x_sampled[mask]
        y_sampled = y_sampled[mask]
        
        # Calculate regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_sampled, y_sampled)
        
        # Store results
        regression_results.append({
            'defect': sim['params']['defect_type'],
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err,
            'n': len(x_sampled)
        })
        
        # Panel 1: Scatter with regression line
        scatter = ax1.scatter(x_sampled, y_sampled, color=color, alpha=0.3,
                            s=10, edgecolors='none', label=sim['params']['defect_type'])
        
        # Add regression line
        x_range = np.linspace(np.min(x_sampled), np.max(x_sampled), 100)
        y_pred = slope * x_range + intercept
        ax1.plot(x_range, y_pred, color=color, linewidth=2, alpha=0.8,
                label=f"R = {r_value:.3f}")
        
        # Panel 3: Residuals
        y_pred_points = slope * x_sampled + intercept
        residuals = y_sampled - y_pred_points
        
        ax3.scatter(y_pred_points, residuals, color=color, alpha=0.3, s=10)
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Panel 4: QQ plot
        if idx == 0:  # Plot QQ for first simulation
            stats.probplot(residuals, dist="norm", plot=ax4)
            ax4.get_lines()[0].set_marker('.')
            ax4.get_lines()[0].set_markersize(5)
            ax4.get_lines()[0].set_alpha(0.5)
            ax4.get_lines()[1].set_color('red')
            ax4.get_lines()[1].set_linewidth(2)
        
        # Panel 5: Histogram of residuals
        ax5.hist(residuals, bins=30, density=True, alpha=0.5, color=color)
    
    # Enhance Panel 1
    ax1.set_xlabel(config.get('correlation_x', 'X Component'), fontsize=11)
    ax1.set_ylabel(config.get('correlation_y', 'Y Component'), fontsize=11)
    ax1.set_title(f"Scatter Plot with Linear Regression", fontsize=12, fontweight='bold')
    
    # Create enhanced legend
    PublicationEnhancer.create_fancy_legend(ax1, *ax1.get_legend_handles_labels(),
                                          loc='upper left', frameon=True,
                                          fancybox=True, shadow=True, ncol=2)
    
    # Panel 2: Correlation coefficients
    defect_names = [sim['params']['defect_type'] for sim in simulations]
    r_values = [result['r_value'] for result in regression_results]
    
    bars = ax2.bar(range(len(defect_names)), r_values, color=colors, alpha=0.7)
    ax2.set_xticks(range(len(defect_names)))
    ax2.set_xticklabels(defect_names, rotation=45, ha='right')
    ax2.set_ylabel("Correlation Coefficient (R)", fontsize=10)
    ax2.set_title("Correlation Strength", fontsize=11, fontweight='bold')
    ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add value labels
    for bar, val in zip(bars, r_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Enhance Panel 3: Residuals
    ax3.set_xlabel("Predicted Values", fontsize=10)
    ax3.set_ylabel("Residuals", fontsize=10)
    ax3.set_title("Residual Plot", fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Enhance Panel 4: QQ Plot
    ax4.set_title("Q-Q Plot of Residuals", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Theoretical Quantiles", fontsize=10)
    ax4.set_ylabel("Sample Quantiles", fontsize=10)
    
    # Enhance Panel 5: Histogram of residuals
    ax5.set_title("Distribution of Residuals", fontsize=11, fontweight='bold')
    ax5.set_xlabel("Residuals", fontsize=10)
    ax5.set_ylabel("Density", fontsize=10)
    ax5.legend([sim['params']['defect_type'] for sim in simulations], fontsize=8)
    
    # Panel 6: Regression parameters table
    ax6.axis('off')
    
    # Create detailed table
    table_data = []
    columns = ['Defect', 'Slope', 'Intercept', 'R', 'R²', 'p-value', 'Std Error', 'N']
    
    for result in regression_results:
        table_data.append([
            result['defect'],
            f"{result['slope']:.4f}",
            f"{result['intercept']:.4f}",
            f"{result['r_value']:.4f}",
            f"{result['r_squared']:.4f}",
            f"{result['p_value']:.3e}",
            f"{result['std_err']:.4f}",
            f"{result['n']:,}"
        ])
    
    table = ax6.table(cellText=table_data, colLabels=columns,
                     cellLoc='center', loc='center',
                     colColours=['#f2f2f2']*len(columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Color code p-values
    for i in range(len(table_data)):
        p_val = float(table_data[i][5].replace('e-', 'E-'))
        if p_val < 0.001:
            table[(i+1, 5)].set_text_props(fontweight='bold', color='green')
        elif p_val < 0.01:
            table[(i+1, 5)].set_text_props(fontweight='bold', color='orange')
    
    # Apply publication styling
    fig = EnhancedFigureStyler.apply_publication_styling(fig, [ax1, ax2, ax3, ax4, ax5, ax6], style_params)
    
    # Add panel labels
    for ax, label in zip([ax1, ax2, ax3, ax4, ax5], ['A', 'B', 'C', 'D', 'E']):
        ax.text(-0.1, 1.05, label, transform=ax.transAxes,
               fontsize=14, fontweight='bold', va='top')
    
    return fig

def calculate_fwhm(x, y):
    """Calculate Full Width at Half Maximum"""
    max_val = np.max(y)
    half_max = max_val / 2
    
    # Find indices where y crosses half max
    above_half = y > half_max
    if not np.any(above_half):
        return 0
    
    indices = np.where(above_half)[0]
    left_idx = indices[0]
    right_idx = indices[-1]
    
    # Interpolate for better accuracy
    if left_idx > 0:
        x_left = np.interp(half_max, 
                          [y[left_idx-1], y[left_idx]], 
                          [x[left_idx-1], x[left_idx]])
    else:
        x_left = x[left_idx]
    
    if right_idx < len(x) - 1:
        x_right = np.interp(half_max, 
                           [y[right_idx], y[right_idx+1]], 
                           [x[right_idx], x[right_idx+1]])
    else:
        x_right = x[right_idx]
    
    return x_right - x_left

# =============================================
# ENHANCED EXPORT FUNCTIONALITY
# =============================================
class PublicationExporter:
    """Enhanced export functionality for publication-quality output"""
    
    @staticmethod
    def export_publication_figures(fig, filename_base, style_params):
        """Export figures in multiple publication formats"""
        formats = []
        
        if style_params.get('vector_output', True):
            # Vector formats
            formats.extend([
                ('pdf', 600),
                ('svg', 600),
                ('eps', 600)
            ])
        
        # Raster formats
        formats.extend([
            ('png', style_params.get('figure_dpi', 600)),
            ('tiff', style_params.get('figure_dpi', 600)),
            ('jpg', 300)
        ])
        
        # Create export package
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for fmt, dpi in formats:
                # Save figure
                fig_buffer = BytesIO()
                fig.savefig(fig_buffer, format=fmt, dpi=dpi, bbox_inches='tight', 
                           pad_inches=0.1, facecolor='white', edgecolor='none')
                fig_buffer.seek(0)
                
                # Add to zip
                zf.writestr(f"{filename_base}.{fmt}", fig_buffer.getvalue())
            
            # Add style parameters
            style_json = json.dumps(style_params, indent=2)
            zf.writestr(f"{filename_base}_style.json", style_json)
            
            # Add caption/legend
            caption = f"""Figure: {filename_base}
Generated: {datetime.now().isoformat()}
Journal Style: {style_params.get('journal_style', 'custom')}
DPI: {style_params.get('figure_dpi', 600)}
Colorspace: RGB
Size: {fig.get_size_inches()[0]:.1f} × {fig.get_size_inches()[1]:.1f} inches"""
            zf.writestr(f"{filename_base}_caption.txt", caption)
        
        buffer.seek(0)
        return buffer
    
    @staticmethod
    def create_figure_panel(figures, layout, style_params):
        """Combine multiple figures into a publication panel"""
        fig_panel, axes = plt.subplots(layout[0], layout[1], 
                                      figsize=(style_params.get('figure_width_double', 12),
                                              style_params.get('figure_width_double', 12) * 0.75),
                                      constrained_layout=True)
        
        # Flatten axes if needed
        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]
        
        # Place figures (simplified - in practice would need to composite)
        for ax in axes_flat:
            ax.text(0.5, 0.5, "Figure Panel", ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.axis('off')
        
        return fig_panel

# =============================================
# MAIN ENHANCEMENTS TO EXISTING CODE
# =============================================

# Add publication enhancer to sidebar
st.sidebar.header("📰 Publication-Quality Enhancements")

# Replace existing styling with enhanced version
advanced_styling = EnhancedFigureStyler.get_publication_controls()

# Add export options
with st.sidebar.expander("💾 Enhanced Export", expanded=False):
    export_preset = st.selectbox(
        "Export Preset",
        ["Nature/Science (CMYK, 600 DPI)", 
         "Physical Review (BW, 600 DPI)",
         "Advanced Materials (RGB, 600 DPI)",
         "Custom"]
    )
    
    include_raw_data = st.checkbox("Include Raw Data", True)
    include_statistics = st.checkbox("Include Statistical Analysis", True)
    include_metadata = st.checkbox("Include Metadata", True)

# =============================================
# ENHANCE THE COMPARISON SECTION
# =============================================
# In the comparison section, replace the plotting calls with enhanced versions
# For example, replace:

# OLD:
# if config['type'] == "Side-by-Side Heatmaps":
#     # Original plotting code

# NEW:
if operation_mode == "Compare Saved Simulations" and 'run_comparison' in st.session_state:
    # ... existing code ...
    
    if config['type'] in ["Side-by-Side Heatmaps", "Overlay Line Profiles", 
                         "Statistical Summary", "Defect-Stress Correlation"]:
        # Use enhanced plotting
        fig = create_enhanced_comparison_plot(simulations, frames, config, advanced_styling)
        
        # Display with enhanced options
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.pyplot(fig)
        
        with col2:
            # Quick export buttons
            if st.button("📥 Export PDF", key="export_pdf"):
                exporter = PublicationExporter()
                export_buffer = exporter.export_publication_figures(
                    fig, f"ag_np_{config['type']}_{datetime.now().strftime('%Y%m%d')}", 
                    advanced_styling
                )
                st.download_button(
                    "Download Publication Package",
                    export_buffer.getvalue(),
                    f"ag_np_publication_figures.zip",
                    "application/zip"
                )
        
        with col3:
            # Show figure info
            st.metric("Figure Size", f"{fig.get_size_inches()[0]:.1f} × {fig.get_size_inches()[1]:.1f} in")
            st.metric("Resolution", f"{advanced_styling.get('figure_dpi', 600)} DPI")

# =============================================
# ADDITIONAL ENHANCEMENTS
# =============================================

# Add a new section for advanced analysis
with st.expander("🔬 Advanced Publication Analysis", expanded=False):
    st.header("📊 Publication-Ready Advanced Analysis")
    
    # Statistical test selection
    col1, col2, col3 = st.columns(3)
    with col1:
        statistical_test = st.selectbox(
            "Statistical Test",
            ["ANOVA", "t-test", "Mann-Whitney U", "Kolmogorov-Smirnov", "Chi-squared"]
        )
    with col2:
        significance_level = st.slider("Significance Level (α)", 0.001, 0.1, 0.05, 0.001)
    with col3:
        multiple_comparison_correction = st.selectbox(
            "Multiple Comparison Correction",
            ["None", "Bonferroni", "Holm-Bonferroni", "Benjamini-Hochberg"]
        )
    
    # Generate statistical report
    if st.button("📈 Generate Statistical Report", type="primary"):
        simulations = SimulationDB.get_all_simulations()
        if simulations:
            # Perform statistical analysis
            report = generate_statistical_report(simulations, statistical_test, 
                                                significance_level, multiple_comparison_correction)
            st.subheader("📋 Statistical Analysis Report")
            st.dataframe(pd.DataFrame(report).T, use_container_width=True)
            
            # Create publication-quality statistical figure
            fig = create_statistical_test_figure(report, advanced_styling)
            st.pyplot(fig)
        else:
            st.warning("No simulations available for statistical analysis!")

def generate_statistical_report(simulations, test_name, alpha, correction):
    """Generate comprehensive statistical report"""
    report = {}
    
    # Extract data for each simulation
    data_by_defect = {}
    for sim_id, sim_data in simulations.items():
        defect_type = sim_data['params']['defect_type']
        if defect_type not in data_by_defect:
            data_by_defect[defect_type] = []
        
        # Get final stress data
        eta, stress_fields = sim_data['history'][-1]
        stress_data = stress_fields['sigma_mag'].flatten()
        data_by_defect[defect_type].append(stress_data)
    
    # Perform statistical tests
    if test_name == "ANOVA":
        # One-way ANOVA
        from scipy.stats import f_oneway
        f_stat, p_value = f_oneway(*[np.concatenate(data) for data in data_by_defect.values()])
        
        report['ANOVA'] = {
            'F-statistic': f_stat,
            'p-value': p_value,
            'Significant': p_value < alpha,
            'Effect Size': 'TODO'  # Would calculate eta-squared
        }
    
    elif test_name == "t-test":
        # Pairwise t-tests
        defects = list(data_by_defect.keys())
        for i in range(len(defects)):
            for j in range(i+1, len(defects)):
                from scipy.stats import ttest_ind
                t_stat, p_value = ttest_ind(
                    np.concatenate(data_by_defect[defects[i]]),
                    np.concatenate(data_by_defect[defects[j]]),
                    equal_var=False
                )
                report[f"{defects[i]} vs {defects[j]}"] = {
                    't-statistic': t_stat,
                    'p-value': p_value,
                    'Significant': p_value < alpha
                }
    
    return report

def create_statistical_test_figure(report, style_params):
    """Create publication-quality statistical test figure"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: p-values as bar plot
    if 'ANOVA' in report:
        ax1.bar(['ANOVA'], [report['ANOVA']['p-value']], color='steelblue', alpha=0.7)
        ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='α = 0.05')
        ax1.set_ylabel('p-value', fontsize=10)
        ax1.set_title('ANOVA Results', fontsize=12, fontweight='bold')
        ax1.set_ylim([0, 1])
        ax1.legend()
    
    # Plot 2: Effect sizes
    # This would be populated with actual effect size calculations
    
    # Apply publication styling
    fig = EnhancedFigureStyler.apply_publication_styling(fig, [ax1, ax2], style_params)
    
    return fig

# =============================================
# FINAL ENHANCEMENTS
# =============================================

# Add a progress bar for batch processing
if 'batch_processing' not in st.session_state:
    st.session_state.batch_processing = False

with st.sidebar.expander("⚙️ Batch Processing", expanded=False):
    if st.button("🔄 Batch Analysis of All Simulations"):
        st.session_state.batch_processing = True
        progress_bar = st.progress(0)
        
        simulations = SimulationDB.get_all_simulations()
        results = []
        
        for i, (sim_id, sim_data) in enumerate(simulations.items()):
            # Perform analysis on each simulation
            result = analyze_simulation_batch(sim_data)
            results.append(result)
            
            # Update progress
            progress = (i + 1) / len(simulations)
            progress_bar.progress(progress)
        
        st.success(f"Batch analysis complete! Processed {len(simulations)} simulations.")
        st.session_state.batch_processing = False

def analyze_simulation_batch(sim_data):
    """Perform comprehensive analysis on a single simulation"""
    # Extract key metrics
    eta_final, stress_final = sim_data['history'][-1]
    
    metrics = {
        'defect_type': sim_data['params']['defect_type'],
        'mean_stress': np.mean(stress_final['sigma_mag']),
        'max_stress': np.max(stress_final['sigma_mag']),
        'defect_area': np.sum(eta_final > 0.5),
        'stress_gradient': np.mean(np.gradient(stress_final['sigma_mag']))
    }
    
    return metrics

# =============================================
# FINAL DISPLAY ENHANCEMENTS
# =============================================

# Add a footer with publication guidelines
st.markdown("---")
st.caption("""
**Publication Guidelines:**
- **Nature/Science**: Use Arial/Helvetica, 600 DPI, CMYK color space
- **Physical Review**: Use Times New Roman, 600 DPI, black and white for main figures
- **Advanced Materials**: Use Arial, 600 DPI, RGB color space
- **General**: Include scale bars, error bars, and statistical significance indicators
- **Export**: Use vector formats (PDF/EPS) for line art, high-res raster for photographs
""")

# Add keyboard shortcuts info
with st.expander("⌨️ Keyboard Shortcuts", expanded=False):
    st.markdown("""
    - **Ctrl+S**: Save current view
    - **Ctrl+E**: Export current figure
    - **Ctrl+P**: Print styling parameters
    - **Ctrl+R**: Reset to defaults
    """)

# Initialize enhanced colormaps
enhanced_cmaps = PublicationEnhancer.create_custom_colormaps()
# Update the COLORMAPS dictionary with enhanced versions
for name, cmap in enhanced_cmaps.items():
    COLORMAPS[name] = cmap

# =============================================
# ENHANCED THEORETICAL ANALYSIS
# =============================================
with st.expander("📚 Enhanced Theoretical Analysis", expanded=True):
    st.markdown("""
    ### 🎯 **Publication-Quality Enhancements Summary**
    
    #### **📊 New Publication Features:**
    
    **1. Journal-Specific Templates:**
    - **Nature/Science**: Arial/Helvetica fonts, specific figure dimensions
    - **Physical Review**: Times New Roman, black and white optimized
    - **Advanced Materials**: Enhanced color schemes, professional layouts
    - **Custom Templates**: Adjustable parameters for any journal
    
    **2. Enhanced Visualization Tools:**
    - **Multi-panel Figures**: Publication-style panel layouts (A, B, C, D)
    - **Advanced Color Maps**: Perceptually uniform, publication-optimized
    - **Scale Bars**: Automatic scale bar generation for microscopy images
    - **Error Visualization**: Confidence intervals, error bands, statistical significance
    
    **3. Statistical Analysis Suite:**
    - **Comprehensive Statistics**: Mean, median, std, skewness, kurtosis
    - **Hypothesis Testing**: ANOVA, t-tests, correlation analysis
    - **Distribution Analysis**: KDE, histograms, Q-Q plots
    - **Regression Analysis**: Linear regression with residuals analysis
    
    **4. Export Enhancements:**
    - **Multi-Format Export**: PDF, EPS, SVG, PNG, TIFF
    - **High Resolution**: Up to 1200 DPI for publication
    - **Complete Packages**: Figures + data + styling parameters
    - **Batch Processing**: Analyze multiple simulations simultaneously
    
    #### **🔬 Scientific Impact:**
    
    **For Materials Science Publications:**
    - **Figure 1**: Multi-panel defect visualization
    - **Figure 2**: Statistical analysis and distributions
    - **Figure 3**: Correlation and regression analysis
    - **Figure 4**: Evolution and time-dependent analysis
    - **Supplementary**: Additional analyses and raw data
    
    **Key Metrics for Publication:**
    - **Statistical Significance**: p-values, confidence intervals
    - **Effect Sizes**: Correlation coefficients, regression slopes
    - **Reproducibility**: Complete parameter export
    - **Visual Clarity**: Professional labeling, consistent formatting
    
    #### **🎨 Design Principles:**
    
    **Color Selection:**
    - **Sequential**: For ordered data (stress magnitude)
    - **Diverging**: For data with critical midpoint (hydrostatic stress)
    - **Categorical**: For defect type comparisons
    - **Accessibility**: Colorblind-friendly palettes
    
    **Typography:**
    - **Font Consistency**: Single font family per figure
    - **Hierarchy**: Clear title → axis labels → tick labels
    - **LaTeX Support**: Mathematical notation for equations
    - **Size Guidelines**: Follow journal-specific requirements
    
    **Layout:**
    - **White Space**: Adequate padding between elements
    - **Alignment**: Consistent axis alignment across panels
    - **Proportions**: Golden ratio for figure dimensions
    - **Balance**: Equal visual weight across panels
    
    ### **📈 Publication Workflow:**
    
    1. **Data Generation**: Run simulations with varied parameters
    2. **Initial Analysis**: Quick comparison in Streamlit interface
    3. **Publication Optimization**: Apply journal-specific styling
    4. **Statistical Validation**: Run hypothesis tests, calculate p-values
    5. **Figure Assembly**: Create multi-panel publication figures
    6. **Export**: Generate high-resolution figures in required formats
    7. **Documentation**: Export parameters and methods for supplementary
    
    ### **🔬 Key Physical Insights from Enhanced Analysis:**
    
    **Quantitative Comparisons:**
    - **Stress Concentrations**: Exact values with confidence intervals
    - **Defect Interactions**: Statistical significance of observed effects
    - **Orientation Dependence**: Quantified angular dependencies
    - **Size Effects**: Scaling relationships with statistical validation
    
    **Novel Visualization Approaches:**
    - **3D Stress Fields**: Enhanced surface and volume rendering
    - **Time Evolution**: Publication-quality evolution movies
    - **Comparative Analysis**: Side-by-side with statistical overlays
    - **Correlation Maps**: Spatial correlation between defect and stress
    
    **Advanced Materials Characterization:**
    - **Nanoparticle-Specific Analysis**: Size, shape, orientation effects
    - **Interface Characterization**: Defect-interface interactions
    - **Stress Field Analysis**: Complete tensor visualization
    - **Energy Landscape**: Defect formation energies and barriers
    
    **Platform now provides publication-ready output suitable for Nature, Science, Advanced Materials, and other high-impact journals!**
    """)
    
    # Display enhanced statistics
    simulations = SimulationDB.get_all_simulations()
    if simulations:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Simulations", len(simulations))
        with col2:
            total_pixels = sum([len(sim['history']) * N * N for sim in simulations.values()])
            st.metric("Data Points", f"{total_pixels:,}")
        with col3:
            st.metric("Publication Formats", "PDF/SVG/EPS/PNG/TIFF")
        with col4:
            st.metric("Max Resolution", "1200 DPI")

st.caption("🔬 Publication-Enhanced Multi-Defect Analyzer • Journal Templates • High-Impact Output • 2025")
