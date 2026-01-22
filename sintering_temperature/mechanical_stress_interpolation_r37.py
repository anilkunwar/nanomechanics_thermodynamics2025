import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# 1. PHYSICS CONSTANTS & MAGNITUDE MAPS
# ==========================================

# Map defect types to a relative magnitude factor.
# This controls "Defect type controlling the magnitude of stress".
# We assume 'Twin' boundaries generate the highest stress, followed by stacking faults.
DEFECT_MAGNITUDE_FACTORS = {
    'ISF': 1.0,      # Intrinsic Stacking Fault
    'ESF': 1.3,      # Extrinsic Stacking Fault
    'Twin': 2.5,      # Twin Boundary (High stress)
    'Void': 1.8,      # Void/Pore
    'Generic': 1.0
}

# ==========================================
# 2. CORE TENSOR ROTATION ENGINE
# ==========================================

class TensorFieldRotator:
    """
    Handles the rotation of 2D Stress Tensor fields.
    
    This enables 'Angular weights controlling orientation' by physically rotating
    the stress tensor components (Sxx, Syy, Sxy) to a new angle.
    """
    
    def __init__(self):
        pass

    @staticmethod
    def rotate_tensor_field(sxx, syy, sxy, angle_rad):
        """
        Rotates a 2D stress tensor field by a specific angle.
        
        Args:
            sxx, syy, sxy: 2D numpy arrays of stress components.
            angle_rad: Rotation angle in radians.
            
        Returns:
            Rotated sxx, syy, sxy arrays.
        """
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        c2 = c * c
        s2 = s * s
        cs = c * s
        
        # Rotation formulas for 2D plane stress:
        # sxx' = sxx*c^2 + syy*s^2 + 2*sxy*s*c
        # syy' = sxx*s^2 + syy*c^2 - 2*sxy*s*c
        # sxy' = (syy - sxx)*s*c + sxy*(c^2 - s^2)
        
        sxx_rot = sxx * c2 + syy * s2 + 2 * sxy * cs
        syy_rot = sxx * s2 + syy * c2 - 2 * sxy * cs
        sxy_rot = (syy - sxx) * cs + sxy * (c2 - s2)
        
        return sxx_rot, syy_rot, sxy_rot

    @staticmethod
    def compute_von_mises(sxx, syy, sxy):
        """Computes Von Mises stress from components."""
        # For plane stress: sigma_vm = sqrt(sxx^2 - sxx*syy + syy^2 + 3*sxy^2)
        return np.sqrt(sxx**2 - sxx * syy + syy**2 + 3 * sxy**2)

# ==========================================
# 3. INTERPOLATION LOGIC
# ==========================================

class RotationalStressInterpolator:
    """
    Interpolates stress fields by rotating source tensors to target orientation
    and scaling magnitudes based on defect type.
    """
    
    def __init__(self, magnitude_factors=None):
        self.rotator = TensorFieldRotator()
        self.magnitude_factors = magnitude_factors if magnitude_factors else DEFECT_MAGNITUDE_FACTORS
        
    def get_angular_weight(self, source_angle, target_angle, sigma=15.0):
        """
        Calculates angular weight using a Gaussian kernel.
        Sources closer to the target angle get higher weights.
        """
        diff = source_angle - target_angle
        # Handle periodicity (360 degree wrap)
        diff = (diff + 180) % 360 - 180
        weight = np.exp(-0.5 * (diff / sigma)**2)
        return weight

    def interpolate(self, sources, target_angle_deg, target_defect_type, grid_shape=(100, 100)):
        """
        Main interpolation driver.
        
        Strategy:
        1. Normalize Source: Divide by its defect magnitude to isolate "Shape/Orientation".
        2. Rotate: Rotate the normalized field to the target angle.
        3. Scale: Multiply by Target Defect Magnitude.
        4. Blend: Weighted average of all transformed sources.
        """
        target_angle_rad = np.radians(target_angle_deg)
        
        # Get target magnitude factor
        target_mag = self.magnitude_factors.get(target_defect_type, 1.0)
        
        # Initialize accumulator for the final interpolated field
        sxx_final = np.zeros(grid_shape)
        syy_final = np.zeros(grid_shape)
        sxy_final = np.zeros(grid_shape)
        total_weight = 0.0
        
        processing_info = []

        for i, source in enumerate(sources):
            # 1. Extract components
            sxx = source['field_sxx']
            syy = source['field_syy']
            sxy = source['field_sxy']
            
            source_angle = source['angle']
            source_type = source['type']
            
            # 2. Magnitude Normalization (Isolate Orientation/Shape)
            source_mag = self.magnitude_factors.get(source_type, 1.0)
            norm_factor = 1.0 / source_mag
            
            sxx_norm = sxx * norm_factor
            syy_norm = syy * norm_factor
            sxy_norm = sxy * norm_factor
            
            # 3. Angular Rotation (Control Orientation)
            # Calculate rotation needed to align source to target
            delta_angle = target_angle_rad - np.radians(source_angle)
            sxx_rot, syy_rot, sxy_rot = self.rotator.rotate_tensor_field(
                sxx_norm, syy_norm, sxy_norm, delta_angle
            )
            
            # 4. Magnitude Injection (Apply Target Intensity)
            sxx_scaled = sxx_rot * target_mag
            syy_scaled = syy_rot * target_mag
            sxy_scaled = sxy_rot * target_mag
            
            # 5. Angular Weighting
            weight = self.get_angular_weight(source_angle, target_angle_deg)
            
            # Accumulate
            sxx_final += weight * sxx_scaled
            syy_final += weight * syy_scaled
            sxy_final += weight * sxy_scaled
            total_weight += weight
            
            processing_info.append({
                "Source ID": i,
                "Angle (deg)": source_angle,
                "Defect": source_type,
                "Rotation (deg)": np.degrees(delta_angle),
                "Weight": weight
            })

        # Normalize by total weight
        if total_weight > 0:
            sxx_final /= total_weight
            syy_final /= total_weight
            sxy_final /= total_weight
        else:
            st.error("Total angular weight is zero. Check angle inputs.")

        # Compute derived scalar fields for visualization
        vm_stress = self.rotator.compute_von_mises(sxx_final, syy_final, sxy_final)
        hydrostatic = (sxx_final + syy_final) / 2.0
        
        return {
            'sxx': sxx_final,
            'syy': syy_final,
            'sxy': sxy_final,
            'von_mises': vm_stress,
            'hydrostatic': hydrostatic,
            'info': processing_info
        }

# ==========================================
# 4. MOCK DATA GENERATOR
# ==========================================

class ShearBandGenerator:
    """
    Generates synthetic 2D stress fields representing a shear band/defect
    oriented at a specific angle.
    """
    @staticmethod
    def generate(grid_size=100, angle_deg=0, defect_type='Twin'):
        x = np.linspace(-1, 1, grid_size)
        y = np.linspace(-1, 1, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # 1. Rotate coordinates to align with defect angle
        theta = np.radians(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        X_rot = X * c + Y * s
        Y_rot = -X * s + Y * c
        
        # 2. Define Stress Distribution (e.g., a shear band)
        # A Gaussian band along the X_rot axis
        
        # Shear stress (Sxy) is high in a shear band
        # Normal stresses (Sxx, Syy) might have concentration at tips or along band
        
        width = 0.15
        intensity = DEFECT_MAGNITUDE_FACTORS.get(defect_type, 1.0) * 1.0 # Base GPa
        
        # Create the "Band"
        band_profile = np.exp(-(Y_rot**2) / (2 * width**2))
        
        # Tensor components for a shear band orientation
        # Simplified physics: Shear dominates along the plane, normal stress perpendicular
        sxx = intensity * 0.2 * band_profile # Compression/Perpendicular
        syy = intensity * 0.2 * band_profile # Compression/Perpendicular
        sxy = intensity * band_profile       # Shear stress along the plane
        
        # Add some noise/roughness
        noise = np.random.normal(0, 0.05 * intensity, (grid_size, grid_size))
        
        # Smooth the fields
        sxx = gaussian_filter(sxx + noise*0.1, sigma=1.0)
        syy = gaussian_filter(syy + noise*0.1, sigma=1.0)
        sxy = gaussian_filter(sxy + noise*0.5, sigma=0.8) # Shear is rougher
        
        return {
            'sxx': sxx,
            'syy': syy,
            'sxy': sxy,
            'angle': angle_deg,
            'type': defect_type
        }

# ==========================================
# 5. VISUALIZATION UTILITIES
# ==========================================

def plot_tensor_field(ax, sxx, syy, sxy, title, cmap='viridis'):
    """Helper to plot tensor components side-by-side."""
    vm = TensorFieldRotator.compute_von_mises(sxx, syy, sxy)
    
    # Plot Von Mises
    im = ax.imshow(vm, cmap=cmap, origin='lower', extent=[-1, 1, -1, 1])
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Add quiver plot to show direction (subsampled)
    step = 5
    Y, X = np.mgrid[-1:1:complex(0, 100), -1:1:complex(0, 100)]
    # Principal stress direction
    # theta = 0.5 * atan(2*sxy / (sxx - syy))
    with np.errstate(divide='ignore', invalid='ignore'):
        principal_angle = 0.5 * np.arctan2(2*sxy, sxx - syy)
    
    U = np.cos(principal_angle)
    V = np.sin(principal_angle)
    
    ax.quiver(X[::step, ::step], Y[::step, ::step], 
              U[::step, ::step], V[::step, ::step], 
              color='white', alpha=0.3, scale=30)
    
    return im

# ==========================================
# 6. MAIN APPLICATION
# ==========================================

def main():
    st.set_page_config(layout="wide", page_title="Rotational Stress Interpolator")
    
    st.title("🔄 Rotational Tensor Field Interpolator")
    st.markdown("""
    This tool interpolates stress fields by **physically rotating** source tensors to match a target orientation.
    *   **Orientation:** Controlled by Angular Weights and Tensor Rotation Matrix.
    *   **Magnitude:** Controlled by Defect Type Mapping (Normalization + Scaling).
    """)
    
    # --- CONFIGURATION ---
    st.sidebar.header("Configuration")
    
    # 1. Define Source Defects (Database)
    st.sidebar.subheader("Source Database (Mock Data)")
    source_configs = [
        {'angle': 0, 'type': 'Twin'},
        {'angle': 45, 'type': 'Twin'},
        {'angle': 90, 'type': 'ISF'},
        {'angle': 135, 'type': 'ESF'},
    ]
    
    # Allow user to modify sources
    edited_sources = []
    for i, cfg in enumerate(source_configs):
        with st.sidebar.expander(f"Source {i+1}"):
            angle = st.slider(f"Angle", 0, 180, cfg['angle'], key=f"src_angle_{i}")
            dtype = st.selectbox(f"Defect Type", list(DEFECT_MAGNITUDE_FACTORS.keys()), 
                                list(DEFECT_MAGNITUDE_FACTORS.keys()).index(cfg['type']), 
                                key=f"src_type_{i}")
            edited_sources.append({'angle': angle, 'type': dtype})
            
    # Generate Data
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Regenerate Database"):
        st.rerun()
        
    # 2. Target Parameters
    st.sidebar.subheader("Target Parameters")
    target_angle = st.slider("Target Orientation (deg)", 0.0, 180.0, 30.0, 0.1)
    target_defect = st.selectbox("Target Defect Type", list(DEFECT_MAGNITUDE_FACTORS.keys()))
    
    # Interpolation Control
    angular_sigma = st.sidebar.slider("Angular Sigma (Interpolation Radius)", 1.0, 90.0, 25.0, 
                                     help="Controls how loosely we blend nearby angles. Low = strict, High = smooth.")
    
    # --- EXECUTION ---
    
    # Step A: Load/Generate Sources
    interpolator = RotationalStressInterpolator()
    source_data = []
    
    for cfg in edited_sources:
        data = ShearBandGenerator.generate(angle_deg=cfg['angle'], defect_type=cfg['type'])
        source_data.append(data)
        
    # Step B: Run Interpolation
    result = interpolator.interpolate(source_data, target_angle, target_defect, sigma=angular_sigma)
    
    # --- VISUALIZATION ---
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Interpolation Result")
        
        # Tabs for different views
        tab_vm, tab_shear, tab_components = st.tabs(["Von Mises (Total Stress)", "Shear Stress (Sxy)", "Tensor Components"])
        
        with tab_vm:
            fig_vm, ax_vm = plt.subplots(figsize=(8, 6))
            im_vm = ax_vm.imshow(result['von_mises'], cmap='inferno', origin='lower')
            ax_vm.set_title(f"Von Mises Stress\nTarget: {target_angle}°, {target_defect}")
            plt.colorbar(im_vm, ax=ax_vm, label="Stress (GPa)")
            st.pyplot(fig_vm)
            
        with tab_shear:
            # Shear stress clearly shows orientation
            fig_shear, ax_shear = plt.subplots(figsize=(8, 6))
            im_shear = ax_shear.imshow(result['sxy'], cmap='RdBu_r', origin='lower', vmin=-2, vmax=2)
            ax_shear.set_title(f"Shear Stress ($\\sigma_{{xy}}$) - Orientation View\nTarget: {target_angle}°")
            plt.colorbar(im_shear, ax=ax_shear, label="Shear (GPa)")
            
            # Overlay a line showing the expected target orientation
            c, s = np.cos(np.radians(target_angle)), np.sin(np.radians(target_angle))
            ax_shear.plot([-c, c], [-s, s], 'k--', linewidth=3, label='Target Orientation')
            ax_shear.legend()
            st.pyplot(fig_shear)
            
        with tab_components:
            fig_comp, axes = plt.subplots(1, 3, figsize=(18, 5))
            plot_tensor_field(axes[0], result['sxx'], result['syy'], result['sxy'], "Normal $\\sigma_{xx}$")
            plot_tensor_field(axes[1], result['sxx'], result['syy'], result['sxy'], "Normal $\\sigma_{yy}$")
            plot_tensor_field(axes[2], result['sxx'], result['syy'], result['sxy'], "Shear $\\sigma_{xy}$")
            st.pyplot(fig_comp)
            
    with col2:
        st.header("Analysis & Weights")
        
        # Show Magnitude Factors used
        st.subheader("Magnitude Control (Defect Type)")
        mag_df = pd.DataFrame(list(DEFECT_MAGNITUDE_FACTORS.items()), columns=["Defect Type", "Factor"])
        st.dataframe(mag_df)
        
        st.markdown(f"""
        *   **Source Normalization:** Each source was divided by its factor.
        *   **Target Scaling:** Result was multiplied by **{target_defect}** factor ({DEFECT_MAGNITUDE_FACTORS[target_defect]}).
        """)
        
        # Show Weighting Breakdown
        st.subheader("Angular Weights")
        df_info = pd.DataFrame(result['info'])
        
        # Bar chart of weights
        fig_weights, ax_w = plt.subplots(figsize=(6, 4))
        colors = ['green' if abs(row['Angle (deg)'] - target_angle) < 5 else 'gray' 
                  for _, row in df_info.iterrows()]
        ax_w.bar(df_info['Source ID'], df_info['Weight'], color=colors)
        ax_w.set_xlabel("Source ID")
        ax_w.set_ylabel("Weight")
        ax_w.set_title("Influence of Each Source")
        st.pyplot(fig_weights)
        
        st.dataframe(df_info, use_container_width=True)
        
        # Comparison with Nearest Neighbor (to prove rotation effect)
        st.subheader("Comparison: Rotated vs. Raw")
        
        # Find closest source
        diffs = [abs(s['angle'] - target_angle) for s in source_data]
        nearest_idx = np.argmin(diffs)
        nearest_src = source_data[nearest_idx]
        
        st.write(f"Nearest Source: {nearest_src['angle']}° ({nearest_src['type']})")
        st.write(f"Target: {target_angle}° ({target_defect})")
        
        # Create a side-by-side comparison of the Von Mises of the nearest source 
        # (rotated but NOT magnitude scaled) vs the full result.
        
        # 1. Nearest Source Rotated only (to isolate rotation effect)
        rot = TensorFieldRotator()
        delta = np.radians(target_angle) - np.radians(nearest_src['angle'])
        ns_sxx_r, ns_syy_r, ns_sxy_r = rot.rotate_tensor_field(nearest_src['sxx'], nearest_src['syy'], nearest_src['sxy'], delta)
        ns_vm_r = rot.compute_von_mises(ns_sxx_r, ns_syy_r, ns_sxy_r)
        
        # 2. Full Result (Rotation + Magnitude Scaling)
        full_vm = result['von_mises']
        
        fig_comp, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(ns_vm_r, cmap='viridis')
        axes[0].set_title("Nearest Source (Rotated Only)")
        axes[1].imshow(full_vm, cmap='viridis')
        axes[1].set_title("Final Result (Rotated + Scaled)")
        st.pyplot(fig_comp)
        
        st.caption("Notice the difference in intensity if the Target Defect Type differs from the Source.")

if __name__ == "__main__":
    main()
