"""
Problem 6 -- Composite (two-layer) cylindrical shell

Core numerical routines: control-volume discretization (general
interior/interface node formula), Thomas-algorithm direct solve,
Jacobi/Gauss-Seidel/SOR iterative solvers, analytical thermal-resistance
solution, and error metrics.

This file defines the "engine" (grid, matrix assembly, solvers).
The driver script run_all.py calls these functions to produce the
tables and plots.
"""

import numpy as np
import time, tracemalloc

# =============================================================================
# Problem data
# -----------------------------------------------------------------------
# All the physical data is collected here in one place, so every function
# below reads from the same source instead of numbers being retyped.
# =============================================================================
r0 = 0.02   # m, radius of the bore (inner) surface, held at fixed temp Ti
r1 = 0.03   # m, radius of the metal/insulation interface
r2 = 0.08   # m, outer radius of the insulation, loses heat by convection
k1 = 45.0   # W/(m K), conductivity of the inner metal (steel) layer
k2 = 0.05   # W/(m K), conductivity of the outer insulation layer
Ti = 150.0  # deg C, fixed (Dirichlet) temperature at the bore, r = r0
Tinf = 25.0 # deg C, ambient fluid temperature outside the pipe
h = 15.0    # W/(m^2 K), convective heat transfer coefficient at r = r2


# =============================================================================
# Analytical thermal-resistance solution
# -----------------------------------------------------------------------
# Closed-form solution using the standard thermal-resistance network
# (like resistances in series in an electrical circuit). Used as the
# exact benchmark to check the finite-difference code against.
# =============================================================================
def analytical_resistances():
    # Conduction resistance of the metal wall (r0 -> r1), per unit length
    R1p = np.log(r1/r0)/(2*np.pi*k1)
    # Conduction resistance of the insulation layer (r1 -> r2), per unit length
    R2p = np.log(r2/r1)/(2*np.pi*k2)
    # Convection resistance at the outer surface (r2 -> ambient), per unit length
    R3p = 1.0/(2*np.pi*r2*h)
    # Total heat flow per unit length = total temperature drop / total resistance
    # (Ohm's law analogy: Q = deltaT / R_total)
    Qp = (Ti - Tinf)/(R1p + R2p + R3p)
    # Temperature at the interface: bore temperature minus the drop across
    # the metal wall only (Q * R1p)
    T_r1 = Ti - Qp*R1p
    # Temperature at the outer surface: interface temperature minus the
    # drop across the insulation layer (Q * R2p)
    T_r2 = T_r1 - Qp*R2p
    return dict(R1p=R1p, R2p=R2p, R3p=R3p, Qp=Qp, T_r1=T_r1, T_r2=T_r2)


def T_exact(r):
    """Analytical temperature at radius r (scalar or array), using the
    thermal-resistance network result. This is the benchmark curve the
    numerical solution is compared against."""
    res = analytical_resistances()
    Qp, T_r1 = res['Qp'], res['T_r1']
    # Accept either a single number or a numpy array of radii
    r = np.atleast_1d(np.asarray(r, dtype=float))
    # The temperature profile is logarithmic in each layer (from solving
    # d/dr(r dT/dr) = 0 analytically): use the metal-layer formula for
    # r <= r1 and the insulation-layer formula for r > r1.
    T = np.where(r <= r1,
                 Ti - Qp/(2*np.pi*k1)*np.log(r/r0),
                 T_r1 - Qp/(2*np.pi*k2)*np.log(np.maximum(r, r1)/r1))
    # If only one point was asked for, return a plain float, not a 1-element array
    return T if T.size > 1 else T.item()


# =============================================================================
# Grid + coefficient assembly
# -----------------------------------------------------------------------
# Build the 1-D radial grid and, for every grid spacing dr, assemble the
# tridiagonal system [A]{T} = {b} coming from the control-volume
# discretization.
# =============================================================================
def build_grid(dr):
    """Build a uniform radial grid r0, r0+dr, ..., r2, making sure the
    metal/insulation interface r1 lands exactly on a grid node (required
    so the interface equation can be applied at a real node)."""
    n1 = int(round((r1 - r0)/dr))   # node index where the interface sits
    n2 = int(round((r2 - r1)/dr))   # number of extra steps to reach r2
    M = n1 + n2                     # index of the last node (nodes are 0..M)
    r = r0 + dr*np.arange(M+1)      # array of all radial node coordinates
    # Sanity checks: make sure rounding didn't shift the interface or the
    # outer boundary off the intended physical location
    assert abs(r[n1] - r1) < 1e-9, "interface not aligned with a grid node"
    assert abs(r[M] - r2) < 1e-9
    return r, n1, M


def k_face(i, n1):
    """Conductivity to use on the face between node i and node i+1
    (i runs from 0 to M-1). If the face lies inside the metal region
    (i < n1) use k1 (steel); otherwise it lies inside the insulation
    region, so use k2. Each face keeps its own material's conductivity --
    k1 and k2 are never averaged."""
    return k1 if i < n1 else k2


def tridiag_coeffs(dr):
    """Build the tridiagonal system A*T = b as three vectors (sub, diag, sup)
    plus the right-hand side b. Row i corresponds to node i (0..M).

    This function assembles all rows of the matrix: the Dirichlet row at
    the bore, the general interior/interface row (the same formula works
    for both), and the Robin (convective) row at the outer surface."""
    r, n1, M = build_grid(dr)
    N = M + 1
    sub = np.zeros(N)   # sub[i] multiplies T_{i-1} (the sub-diagonal); unused at i=0
    diag = np.zeros(N)  # diag[i] multiplies T_i (the main diagonal)
    sup = np.zeros(N)   # sup[i] multiplies T_{i+1} (the super-diagonal); unused at i=M
    b = np.zeros(N)     # right-hand-side vector

    # --- Row i = 0 : Dirichlet boundary condition at the bore -------------
    # Forces T[0] = Ti, so this row of the matrix is trivial: 1*T0 = Ti.
    diag[0] = 1.0
    b[0] = Ti

    # --- Rows i = 1 .. M-1 : interior nodes AND the interface node --------
    # One general control-volume formula is used here. It reduces to the
    # plain interior-node equation when both neighbouring faces share the
    # same material, and automatically enforces flux continuity with the
    # correct (non-averaged) conductivities at the node where the two
    # materials meet.
    for i in range(1, M):
        kL = k_face(i-1, n1)      # conductivity of the face to the left  (i-1 -> i)
        kR = k_face(i, n1)        # conductivity of the face to the right (i -> i+1)
        r_mh = r[i] - dr/2.0      # r_{i-1/2}: radius at the midpoint of the left face
        r_ph = r[i] + dr/2.0      # r_{i+1/2}: radius at the midpoint of the right face
        # These coefficients come from integrating the governing equation
        # d/dr(r dT/dr) = 0 over the control volume around node i, using
        # k*r/dr as the "conductance" of each face.
        cL = kL*r_mh
        cR = kR*r_ph
        sub[i] = cL
        sup[i] = cR
        diag[i] = -(cL + cR)      # energy balance: what comes in must equal what goes out
        b[i] = 0.0                # no internal heat generation in this problem

    # --- Row i = M : Robin (convective) boundary at the outer surface -----
    # The flux conducted from the last interior half-cell must equal the
    # convective heat loss to the ambient fluid, h*(T_M - Tinf), over the
    # outer circumference (per unit length, so the "area" term is 2*pi*r2
    # with the 2*pi cancelled on both sides).
    kL = k_face(M-1, n1)
    r_mh = r[M] - dr/2.0
    cL = kL*r_mh/dr
    diag[M] = -(cL + h*r2)
    sub[M] = cL
    b[M] = -h*r2*Tinf

    return sub, diag, sup, b, r, n1, M


# =============================================================================
# Direct solve: Thomas algorithm (tridiagonal Gaussian elimination)
# -----------------------------------------------------------------------
# Because this is a strictly 1-D problem, [A] is exactly tridiagonal with
# no fill-in, so the Thomas algorithm (a specialised, cheap form of
# Gaussian elimination) is the natural direct method -- O(M) time and
# memory instead of O(M^3)/O(M^2) for a dense solve.
# =============================================================================
def thomas_solve(sub, diag, sup, b):
    """Standard forward-elimination / back-substitution Thomas algorithm
    for a tridiagonal system. sub, diag, sup, b are the three diagonals
    and the right-hand side produced by tridiag_coeffs()."""
    N = len(diag)
    cp = np.zeros(N)   # modified super-diagonal coefficients (forward sweep)
    dp = np.zeros(N)   # modified right-hand-side values (forward sweep)

    # Forward elimination: normalise the first row, then eliminate the
    # sub-diagonal entry of every subsequent row using the row above it.
    cp[0] = sup[0]/diag[0]
    dp[0] = b[0]/diag[0]
    for i in range(1, N):
        m = diag[i] - sub[i]*cp[i-1]        # pivot after eliminating sub[i]
        cp[i] = sup[i]/m if i < N-1 else 0.0
        dp[i] = (b[i] - sub[i]*dp[i-1])/m

    # Back substitution: start from the last node (already solved) and
    # work backwards to fill in every other node.
    x = np.zeros(N)
    x[-1] = dp[-1]
    for i in range(N-2, -1, -1):
        x[i] = dp[i] - cp[i]*x[i+1]
    return x


def solve_direct(dr):
    """Assemble the system for grid spacing dr and solve it with the
    Thomas algorithm, timing the run and measuring peak memory."""
    tracemalloc.start()             # start tracking memory allocations
    t0 = time.perf_counter()        # start the wall-clock timer

    sub, diag, sup, b, r, n1, M = tridiag_coeffs(dr)
    T = thomas_solve(sub, diag, sup, b)

    t1 = time.perf_counter()
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    info = dict(time=t1-t0, peak_mem_kb=peak/1024, N=len(diag))
    return T, r, n1, M, info


# =============================================================================
# Iterative solvers (Jacobi / Gauss-Seidel / SOR) on the SAME tridiagonal
# system
# -----------------------------------------------------------------------
# These solve the exact same [A]{T} = {b} system built above, but by
# repeated sweeps instead of direct elimination, so direct vs. iterative
# performance can be compared.
# =============================================================================
def iterative_solve(dr, method='gs', omega=1.0, tol=1e-6, maxiter=2_000_000):
    """method: 'jacobi' for Jacobi, 'gs' for Gauss-Seidel (omega=1), or
    'sor' for SOR with the given relaxation factor omega.
    tol: convergence tolerance on the max change between sweeps.
    """
    sub, diag, sup, b, r, n1, M = tridiag_coeffs(dr)
    N = len(diag)

    # Initial guess: a straight line between the bore temperature and
    # ambient temperature.
    T = np.linspace(Ti, Tinf, N)

    tracemalloc.start()
    t0 = time.perf_counter()
    n_iter = 0

    for it in range(1, maxiter+1):
        T_old = T.copy()   # keep the previous sweep's values to check convergence

        if method == 'jacobi':
            # Jacobi: every node is updated using ONLY last sweep's values
            # (T_old), so all updates for this sweep are independent of
            # each other and could, in principle, be done in parallel.
            T_new = T.copy()
            for i in range(N):
                s = b[i]
                if i > 0:
                    s -= sub[i]*T_old[i-1]
                if i < N-1:
                    s -= sup[i]*T_old[i+1]
                T_new[i] = s/diag[i]
            T = T_new
        else:
            # Gauss-Seidel / SOR: sweep left to right and use the freshly
            # updated neighbour values as soon as they're available -- this
            # is why it converges faster than Jacobi. Since the system is a
            # simple 1-D chain, an ordinary sequential sweep (no
            # red-black colouring) is enough.
            for i in range(N):
                s = b[i]
                if i > 0:
                    s -= sub[i]*T[i-1]
                if i < N-1:
                    s -= sup[i]*T[i+1]
                gs_val = s/diag[i]
                # SOR over-relaxation step: push the update further than
                # plain Gauss-Seidel would (omega=1 reduces exactly to
                # Gauss-Seidel; omega>1 is over-relaxation).
                T[i] = T[i] + omega*(gs_val - T[i])

        # Convergence check: stop once the largest change anywhere in the
        # domain, between this sweep and the last, drops below tol.
        diff = np.max(np.abs(T - T_old))
        n_iter = it
        if diff < tol:
            break

    t1 = time.perf_counter()
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    converged = diff < tol
    info = dict(iterations=n_iter, time=t1-t0, peak_mem_kb=peak/1024,
                converged=converged, final_diff=diff, N=N)
    return T, r, n1, M, info


# =============================================================================
# Error metrics
# -----------------------------------------------------------------------
# Helper functions used to compare a numerical solution against the
# analytical benchmark.
# =============================================================================
def compare_to_exact(T, r):
    """Return the exact temperature, absolute error, and percentage error
    at every grid point, for a full-field comparison."""
    Te = T_exact(r)
    err = np.abs(T - Te)
    pct = 100*err/np.abs(Te)
    return Te, err, pct


def max_error(T, r):
    """Single number: the worst-case (maximum) absolute error over the
    whole domain -- tracked through the grid-refinement study."""
    Te = T_exact(r)
    return np.max(np.abs(T - Te))
