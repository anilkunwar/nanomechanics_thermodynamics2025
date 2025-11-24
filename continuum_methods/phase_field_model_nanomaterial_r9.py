import numpy as np
from numba import jit, prange

# Parameters
N = 64  # Grid size (increase for resolution, but memory-heavy)
dx = 0.1  # nm/grid
dt = 0.005
steps = 50
M = 1.0
kappa = 0.5
C44 = 46.1
eps0 = 0.707  # ISF-like; change to 1.414 for ESF, 2.121 for Twin

# 3D Grid
x = np.linspace(-N*dx/2, N*dx/2, N)
X, Y, Z = np.meshgrid(x, x, x)
r = np.sqrt(X**2 + Y**2 + Z**2)
np_radius = N*dx/2 * 0.8  # Spherical nanoparticle radius

# Initial eta (spherical inclusion defect at center)
eta = np.zeros((N, N, N))
inclusion_mask = r < np_radius / 3  # Small central defect
eta[inclusion_mask] = 0.5  # Seed value

@jit(nopython=True, parallel=True)
def evolve_3d(eta, kappa, dt, dx, N):
    eta_new = eta.copy()
    dx2 = dx**2
    for i in prange(1, N-1):
        for j in prange(1, N-1):
            for k in prange(1, N-1):
                lap = (eta[i+1,j,k] + eta[i-1,j,k] +
                       eta[i,j+1,k] + eta[i,j-1,k] +
                       eta[i,j,k+1] + eta[i,j,k-1] - 6*eta[i,j,k]) / dx2
                dF = 2*eta[i,j,k]*(1-eta[i,j,k])*(eta[i,j,k]-0.5)
                eta_new[i,j,k] = eta[i,j,k] + dt * (-dF + kappa * lap)
                eta_new[i,j,k] = max(0.0, min(1.0, eta_new[i,j,k]))
    # Periodic BC
    eta_new[0,:,:] = eta_new[-2,:,:]
    eta_new[-1,:,:] = eta_new[1,:,:]
    eta_new[:,0,:] = eta_new[:,-2,:]
    eta_new[:,-1,:] = eta_new[:,1,:]
    eta_new[:,:,0] = eta_new[:,:,-2]
    eta_new[:,:,-1] = eta_new[:,:,1]
    return eta_new

def compute_stress_3d(eta, eps0, C44):
    eps_star = eps0 * eta
    eps_fft = np.fft.fftn(eps_star)
    kx = np.fft.fftfreq(N, dx) * 2*np.pi
    ky = kx.copy()
    kz = kx.copy()
    KX, KY, KZ = np.meshgrid(kx, ky, kz)
    k2 = KX**2 + KY**2 + KZ**2 + 1e-12
    strain_fft = -eps_fft / (2 * k2)
    strain = np.real(np.fft.ifftn(strain_fft))
    sigma = C44 * strain
    return sigma

# Run evolution
for step in range(steps):
    eta = evolve_3d(eta, kappa, dt, dx, N)
    sigma = compute_stress_3d(eta, eps0, C44)

# Mask to spherical NP
mask_np = r <= np_radius
eta[~mask_np] = 0
sigma[~mask_np] = 0

# Save VTU (manual XML – for ParaView)
with open("np_3d.vtu", 'w') as f:
    f.write('<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">\n')
    f.write(f'<StructuredGrid WholeExtent="0 {N-1} 0 {N-1} 0 {N-1}">\n')
    f.write(f'<Piece Extent="0 {N-1} 0 {N-1} 0 {N-1}">\n')
    f.write('<PointData Scalars="fields">\n')
    f.write('<DataArray type="Float32" Name="eta" NumberOfComponents="1" format="ascii">\n')
    f.write(' '.join([str(v) for v in eta.flatten('F')]))
    f.write('\n</DataArray>\n')
    f.write('<DataArray type="Float32" Name="sigma" NumberOfComponents="1" format="ascii">\n')
    f.write(' '.join([str(v) for v in sigma.flatten('F')]))
    f.write('\n</DataArray>\n')
    f.write('</PointData>\n')
    f.write('</Piece>\n</StructuredGrid>\n</VTKFile>')

print("Simulation complete – np_3d.vtu saved")
print("Max eta:", np.max(eta))
print("Max sigma:", np.max(sigma))
