"""
Steady 2D conduction in a solid cylinder (polar coordinates).
Core numerical routines: matrix assembly, direct solve, Jacobi/GS/SOR (red-black),
analytical solution, error metrics, boundary heat flux.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time, tracemalloc

# Problem data: cylinder radius, boundary-condition amplitudes, and
# thermal conductivity.
R      = 0.08   # outer radius of the cylinder [m]
T0d    = 80.0   # mean (DC) temperature on the boundary [deg C]
dT1    = 25.0   # amplitude of the cos(theta) boundary component [deg C]
dT2    = 10.0   # amplitude of the cos(2 theta) boundary component [deg C]
k      = 15.0   # thermal conductivity [W/(m K)], used only for the flux calc

def Tb(theta):
    """Prescribed Dirichlet boundary temperature T(R, theta)."""
    return T0d + dT1*np.cos(theta) + dT2*np.cos(2*theta)

def T_exact(r, theta):
    """
    Analytical (separation-of-variables) solution T(r, theta). Satisfies
    Laplace's equation, stays bounded at r=0, and matches the boundary
    condition above -- used as a benchmark for the numerical solution.
    """
    return T0d + dT1*(r/R)*np.cos(theta) + dT2*(r/R)**2*np.cos(2*theta)

def idx(i, j, N):
    """
    Map a 2-D grid node (ring i, angular position j) to its 1-D position in
    the global unknown vector. Index 0 is reserved for the single pole
    unknown (r=0); interior ring nodes i=1..M-1 are laid out with the
    angular index j varying fastest, giving Ntot = 1 + (M-1)*N unknowns.
    """
    return 1 + (i-1)*N + j

def unknown_count(M, N):
    """Total number of unknowns: 1 pole node + (M-1) interior rings of N nodes each."""
    return 1 + (M-1)*N

def assemble(M, N):
    """
    Build the sparse linear system [A]{T} = {b} for the interior 5-point
    polar stencil, plus the pole averaging equation. Uses a uniform grid:
    dr = R/M in the radial direction and dtheta = 2*pi/N in the angular
    direction (periodic in theta).
    """
    dr = R/M
    dtheta = 2*np.pi/N
    theta = dtheta*np.arange(N)          # angular positions of ring nodes
    Ntot = unknown_count(M, N)

    # Build the sparse matrix in COO format (parallel row/col/val lists)
    # and convert to CSR at the end -- much cheaper than assigning into
    # a dense or already-CSR matrix entry by entry.
    rows, cols, vals = [], [], []
    b = np.zeros(Ntot)

    # Pole equation: T0 = (1/N) * sum_j T(1,j), i.e. the centre temperature
    # equals the average of the surrounding first-ring nodes. Written as a
    # homogeneous row: T0 - (1/N)*sum_j T_{1,j} = 0.
    rows.append(0); cols.append(0); vals.append(1.0)
    for j in range(N):
        rows.append(0); cols.append(idx(1, j, N)); vals.append(-1.0/N)
    b[0] = 0.0

    # Interior ring nodes i = 1..M-1: standard 5-point polar Laplacian stencil.
    for i in range(1, M):
        r_i = i*dr
        # Radial and angular stencil coefficients depend only on the ring
        # radius r_i, not on theta.
        a = 1.0/dr**2 + 1.0/(2*r_i*dr)       # coefficient of outward neighbour T_{i+1,j}
        c_ = 1.0/dr**2 - 1.0/(2*r_i*dr)      # coefficient of inward neighbour T_{i-1,j}
        cth = 1.0/(r_i**2 * dtheta**2)       # coefficient of each angular neighbour
        center = -(2.0/dr**2 + 2.0*cth)      # coefficient of the node itself

        for j in range(N):
            k_row = idx(i, j, N)
            # Periodicity in theta: angular neighbours wrap around mod N,
            # so the last column and first column are coupled (this is
            # what makes each angular block "cyclic tridiagonal" rather
            # than plain tridiagonal).
            jp = (j+1) % N
            jm = (j-1) % N

            rows.append(k_row); cols.append(k_row); vals.append(center)
            rows.append(k_row); cols.append(idx(i, jp, N)); vals.append(cth)
            rows.append(k_row); cols.append(idx(i, jm, N)); vals.append(cth)

            # Outward neighbour: either the next interior ring, or (if this
            # is the last interior ring, i+1==M) the known Dirichlet
            # boundary value, which is moved to the right-hand side b.
            if i+1 == M:
                b[k_row] -= a*Tb(theta[j])
            else:
                rows.append(k_row); cols.append(idx(i+1, j, N)); vals.append(a)

            # Inward neighbour: either the previous interior ring, or (if
            # this is the first interior ring, i-1==0) the single pole
            # unknown T0, common to every angular position.
            if i-1 == 0:
                rows.append(k_row); cols.append(0); vals.append(c_)
            else:
                rows.append(k_row); cols.append(idx(i-1, j, N)); vals.append(c_)

    # Assemble the sparse matrix from the collected (row, col, value) triples.
    A = sp.csr_matrix((vals, (rows, cols)), shape=(Ntot, Ntot))
    return A, b, dr, dtheta, theta

def unpack(Tvec, M, N):
    """
    Convert the 1-D solution vector Tvec (pole + interior ring unknowns)
    back into a 2-D array T[i, j] of shape (M+1, N), filling in the pole
    row (i=0, same value at every theta) and the known Dirichlet boundary
    row (i=M) directly from Tb().
    """
    dtheta = 2*np.pi/N
    theta = dtheta*np.arange(N)
    T = np.zeros((M+1, N))
    T[0, :] = Tvec[0]                     # pole temperature, same for all j
    for i in range(1, M):
        for j in range(N):
            T[i, j] = Tvec[idx(i, j, N)]
    T[M, :] = Tb(theta)                   # outer boundary, known analytically
    return T

def solve_direct(M, N):
    """
    Direct solve of [A]{T} = {b} via sparse LU (SuperLU through
    scipy.sparse.linalg.splu), timed and memory-profiled with
    tracemalloc for the grid-refinement performance study.
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    A, b, dr, dtheta, theta = assemble(M, N)
    lu = spla.splu(A.tocsc())             # SuperLU needs CSC format
    Tvec = lu.solve(b)
    t1 = time.perf_counter()
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    nnz = A.nnz
    Ntot = A.shape[0]
    return unpack(Tvec, M, N), dict(time=t1-t0, peak_mem_kb=peak/1024,
                                     nnz=nnz, Ntot=Ntot, bandwidth_naive=N+1)

def max_error_grid(T, M, N):
    """
    Maximum absolute error over the entire computed field T[i,j] versus
    the analytical solution T_exact, evaluated at every grid point.
    Used for the grid-convergence study.
    """
    dr = R/M
    dtheta = 2*np.pi/N
    i_idx = np.arange(0, M+1)
    j_idx = np.arange(0, N)
    II, JJ = np.meshgrid(i_idx, j_idx, indexing='ij')
    Rg = II*dr
    THg = JJ*dtheta
    Te = T_exact(Rg, THg)
    err = np.abs(T - Te)
    return np.max(err)

def surface_flux(T, M, N):
    """
    Local outward radial heat flux q''(theta) = -k dT/dr at r=R, computed
    with a second-order one-sided three-point finite difference using the
    two innermost neighbouring rings (M-1, M-2) since a centred difference
    isn't available right at the boundary.
    """
    dr = R/M
    dTdr = (3*T[M, :] - 4*T[M-1, :] + T[M-2, :]) / (2*dr)
    q = -k*dTdr
    return q

def flux_integral_check(T, M, N):
    """
    Energy-balance check: with no internal heat generation, the net flux
    integrated around the circumference should vanish,
    integral_0^2pi q''(theta) R dtheta ~= 0. Approximated here by a
    simple Riemann sum over the N angular flux samples.
    """
    dtheta = 2*np.pi/N
    q = surface_flux(T, M, N)
    integral = np.sum(q)*R*dtheta
    return integral, q
