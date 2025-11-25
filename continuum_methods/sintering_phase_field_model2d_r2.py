import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =============================================
# 2D Phase-Field Model for Multi-Particle Sintering
# =============================================
st.set_page_config(page_title="2D Multi-Particle Sintering Simulator", layout="wide")
st.title("2D Phase-Field Model for Multi-Particle Sintering in Silver Nanoparticles")
st.markdown("""
Allen-Cahn based simulation of nanoparticle fusion.  
Particles placed randomly but close for neck formation.  
Metric: Increase in 'neck' pixels (0.1 < φ < 0.9).
""")

# Parameters (user editable)
n_particles = st.sidebar.slider("Number of particles (n)", 2, 10, 5)
grid_size = st.sidebar.slider("Grid size (N x N)", 100, 400, 200, 50)
steps = st.sidebar.slider("Simulation steps", 100, 1000, 500, 50)
dt = st.sidebar.slider("Time step dt", 0.005, 0.05, 0.01, 0.005)
M = st.sidebar.slider("Mobility M", 0.5, 2.0, 1.0, 0.1)
kappa = st.sidebar.slider("Gradient coeff κ", 0.5, 2.0, 1.0, 0.1)
min_radius = st.sidebar.slider("Min particle radius", 10, 30, 20)
max_radius = st.sidebar.slider("Max particle radius", min_radius, 40, 25)
save_every = st.sidebar.slider("Save every n steps", 10, 100, 50)

# Initialize phase field φ
N = grid_size
dx = 1.0
phi = np.zeros((N, N))

# Random close placement without too much overlap
np.random.seed(42)
centers = []
radii = []
while len(centers) < n_particles:
    radius = np.random.uniform(min_radius, max_radius)
    overlap = False  # Moved outside if-else
    if len(centers) == 0:
        cx = np.random.uniform(radius, N - radius)
        cy = np.random.uniform(radius, N - radius)
    else:
        # Place near random previous particle
        idx = np.random.randint(0, len(centers))
        prev_cx, prev_cy = centers[idx]
        prev_r = radii[idx]
        angle = np.random.uniform(0, 2*np.pi)
        dist = prev_r + radius - np.random.uniform(0, 10)  # Allow gap/overlap for sintering
        cx = prev_cx + dist * np.cos(angle)
        cy = prev_cy + dist * np.sin(angle)
        cx = np.clip(cx, radius, N - radius)
        cy = np.clip(cy, radius, N - radius)
        
        # Check excessive overlap
        for pcx, pcy, pr in zip([c[0] for c in centers], [c[1] for c in centers], radii):
            pdist = np.sqrt((cx - pcx)**2 + (cy - pcy)**2)
            if pdist < pr + radius - 15:  # Allow close but not too merged
                overlap = True
                break
    if not overlap:
        centers.append((cx, cy))
        radii.append(radius)
        y, x = np.ogrid[0:N, 0:N]
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        phi[mask] = 1.0

phi_initial = phi.copy()

# Laplacian function
def laplacian(phi):
    lap = (np.roll(phi, 1, 0) + np.roll(phi, -1, 0) +
           np.roll(phi, 1, 1) + np.roll(phi, -1, 1) - 4 * phi) / dx**2
    return lap

# Simulate (Allen-Cahn for non-conserved interface evolution)
for step in range(steps):
    mu = 4 * phi * (1 - phi) * (phi - 0.5) - kappa * laplacian(phi)
    phi += dt * M * mu
    phi = np.clip(phi, 0, 1)

# Sintering metric: increase in 'neck' regions (0.1 < phi < 0.9)
initial_neck = np.sum((phi_initial > 0.1) & (phi_initial < 0.9))
final_neck = np.sum((phi > 0.1) & (phi < 0.9))
metric = final_neck - initial_neck

# Display results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(phi_initial, cmap='gray')
ax1.set_title('Initial Configuration')
ax2.imshow(phi, cmap='gray')
ax2.set_title('After Sintering')
st.pyplot(fig)

st.success(f"Sintering metric (neck pixel increase): {metric}")
st.caption("2D Allen-Cahn phase-field model • Random close particles • Neck formation via interface minimization • Ag NPs 2025")
