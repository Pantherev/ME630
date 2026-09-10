import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend, so plots save to file without a display
import matplotlib.pyplot as plt
import json, csv, os, time
from solver import (R, T0d, dT1, dT2, k, Tb, T_exact, assemble, solve_direct,
                     max_error_grid, surface_flux, flux_integral_check, unknown_count)

OUT = 'out'
os.makedirs(OUT, exist_ok=True)

# Grid levels for the refinement study: M (radial) and N (angular) are
# doubled together at each level so dr and dtheta refine at the same rate.
grids = [(16, 32), (32, 64), (64, 128), (128, 256)]

# ----------------------------------------------------------------------
# Iterative solvers (Jacobi / red-black GS / red-black SOR)
# ----------------------------------------------------------------------
def _coeffs(M, N):
    """
    Precompute the radius-dependent stencil coefficients for every interior
    ring i=1..M-1, as 1-D arrays indexed by i. Reused every sweep of the
    iterative solvers below instead of recomputing per node.
    """
    dr = R/M
    dtheta = 2*np.pi/N
    i_arr = np.arange(1, M)
    r_i = i_arr*dr
    a = 1.0/dr**2 + 1.0/(2*r_i*dr)        # outward-neighbour coefficient
    c_ = 1.0/dr**2 - 1.0/(2*r_i*dr)       # inward-neighbour coefficient
    cth = 1.0/(r_i**2*dtheta**2)          # angular-neighbour coefficient
    center = 2.0/dr**2 + 2.0*cth          # (positive) diagonal coefficient
    return dr, dtheta, a, c_, cth, center

def _init_field(M, N):
    """
    Initial guess for the iterative solvers: linear interpolation in the
    radial direction between a uniform pole guess T0d and the known
    boundary profile Tb(theta).
    """
    dtheta = 2*np.pi/N
    theta = dtheta*np.arange(N)
    T = np.zeros((M+1, N))
    Tb_row = Tb(theta)
    for i in range(M+1):
        frac = i/M
        T[i, :] = (1-frac)*T0d + frac*Tb_row
    return T

def iterative_solve(M, N, method='jacobi', omega=1.0, tol=1e-6, maxiter=60000):
    """
    Solve the same discretized system as assemble()/solve_direct(), but by
    fixed-point iteration instead of direct LU factorization.

    method='jacobi' : plain (unrelaxed) Jacobi sweep.
    method='gs'      : Gauss-Seidel via red-black (checkerboard) ordering
                        with omega=1 (no relaxation).
    method='sor'     : same red-black sweep but with relaxation factor
                        omega, i.e. SOR (omega=1 reduces to GS).

    Red-black ordering lets each colour's update be done as one vectorised
    NumPy operation over the whole field (instead of a Python loop over
    every node), while still using the freshest neighbour values within a
    sweep the way classical Gauss-Seidel does.
    """
    dr, dtheta, a, c_, cth, center = _coeffs(M, N)
    T = _init_field(M, N)
    Mm1 = M-1
    theta = dtheta*np.arange(N)
    Tb_row = Tb(theta)
    T[M, :] = Tb_row                       # outer Dirichlet boundary, fixed for all iterations

    # Checkerboard (red/black) masks over the interior (ring, angle) grid,
    # based on parity of i+j, used to update alternating subsets of nodes.
    I, J = np.meshgrid(np.arange(Mm1), np.arange(N), indexing='ij')
    red_mask = ((I+J) % 2 == 0)
    black_mask = ~red_mask

    # Reshape 1-D radial coefficient arrays into column vectors so they
    # broadcast correctly against the (ring x angle) 2-D field arrays below.
    a_col = a.reshape(-1, 1)
    c_col = c_.reshape(-1, 1)
    cth_col = cth.reshape(-1, 1)
    center_col = center.reshape(-1, 1)

    import tracemalloc
    tracemalloc.start()
    t0 = time.perf_counter()
    n_iter = 0
    diff = None
    for it in range(1, maxiter+1):
        T_old_full = T.copy()             # kept for the convergence-check diff below

        if method == 'jacobi':
            # Plain Jacobi: every new value uses only OLD neighbour values,
            # so the whole interior can be updated in a single vectorised pass.
            interior = T[1:M, :]
            up = T[2:M+1, :]              # outward neighbours (ring i+1)
            down = np.empty_like(interior)
            down[0, :] = T[0, 0]          # innermost ring's "inward" neighbour is the pole
            if M-1 > 1:
                down[1:, :] = T[1:M-1, :] # inward neighbours (ring i-1) for i>1
            left = np.roll(interior, 1, axis=1)   # angular neighbour j-1 (periodic wrap)
            right = np.roll(interior, -1, axis=1) # angular neighbour j+1 (periodic wrap)
            new_interior = (a_col*up + c_col*down + cth_col*(left+right)) / center_col
            T[1:M, :] = new_interior
            T[0, :] = np.mean(T[1, :])    # pole update: average of first-ring nodes
        else:
            # Gauss-Seidel (omega=1) or SOR (omega != 1), applied colour by
            # colour so that the black-node update sees the just-updated
            # red nodes, mimicking sequential GS while staying vectorised.
            for mask in (red_mask, black_mask):
                interior = T[1:M, :]
                up = T[2:M+1, :]
                down = np.empty_like(interior)
                down[0, :] = T[0, 0]
                if M-1 > 1:
                    down[1:, :] = T[1:M-1, :]
                left = np.roll(interior, 1, axis=1)
                right = np.roll(interior, -1, axis=1)
                gs_val = (a_col*up + c_col*down + cth_col*(left+right)) / center_col
                # SOR relaxation: new = old + omega*(GS_value - old), applied
                # only to nodes of the current colour (mask); the other
                # colour's nodes pass through unchanged this half-sweep.
                updated = np.where(mask, T[1:M, :] + omega*(gs_val - T[1:M, :]), T[1:M, :])
                T[1:M, :] = updated
            # Pole update is likewise relaxed by the same omega.
            gs_pole = np.mean(T[1, :])
            T[0, :] = T[0, 0] + omega*(gs_pole - T[0, 0])

        # Convergence check (all methods): stop once the largest change
        # over the whole field between sweeps drops below tol.
        diff = np.max(np.abs(T - T_old_full))
        n_iter = it
        if diff < tol:
            break

    t1 = time.perf_counter()
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    converged = diff < tol
    return T, dict(iterations=n_iter, time=t1-t0, peak_mem_kb=peak/1024,
                   converged=converged, final_diff=diff)

def compare_to_exact(T, M, N, r_fracs=(0.25, 0.5, 0.75, 1.0), theta_deg=(0, 45, 90, 180)):
    """
    Point-wise comparison: sample the computed field T at the requested
    fractional radii and angles, and report absolute/percentage error
    against the analytical T_exact at each sample point.
    """
    dr = R/M
    dtheta = 2*np.pi/N
    rows = []
    for rf in r_fracs:
        i = int(round(rf*M))
        r_val = i*dr
        for th in theta_deg:
            j = int(round(th/(360.0/N))) % N
            th_actual = j*dtheta
            Tn = T[i, j]
            Te = T_exact(r_val, th_actual)
            err = Tn - Te
            pct = 100*abs(err)/abs(Te) if abs(Te) > 1e-12 else 0.0
            rows.append(dict(r_frac=rf, r=r_val, theta_deg=th, Tnum=Tn, Texact=Te,
                              abs_err=abs(err), pct_err=pct))
    return rows

# =====================================================================
# 1) Direct solve + grid refinement study
# =====================================================================
# For every grid level: solve directly, record system size/sparsity/time/
# memory, and compute the max error against the analytical solution.
direct_results = {}
refinement_rows = []
for (M, N) in grids:
    T, info = solve_direct(M, N)
    direct_results[(M, N)] = (T, info)
    maxerr = max_error_grid(T, M, N)
    dr = R/M
    refinement_rows.append(dict(M=M, N=N, dr=dr, dtheta=2*np.pi/N,
                                 Ntot=info['Ntot'], nnz=info['nnz'],
                                 time_s=info['time'], mem_kb=info['peak_mem_kb'],
                                 max_abs_err=maxerr))
    print(f"Direct M={M} N={N}: Ntot={info['Ntot']} nnz={info['nnz']} "
          f"time={info['time']*1e3:.3f} ms  mem={info['peak_mem_kb']:.1f} kB "
          f"max|err|={maxerr:.3e}")

with open(f'{OUT}/refinement_direct.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(refinement_rows[0].keys()))
    w.writeheader()
    w.writerows(refinement_rows)

# Observed order of accuracy p, estimated between consecutive grid levels
# from p = log(err_coarse/err_fine) / log(dr_coarse/dr_fine); should be
# close to 2 for the second-order 5-point stencil.
print("\nObserved order of accuracy (direct solve, based on max abs error):")
orders = []
for a, b_ in zip(refinement_rows[:-1], refinement_rows[1:]):
    order = np.log(a['max_abs_err']/b_['max_abs_err']) / np.log(a['dr']/b_['dr'])
    orders.append(order)
    print(f"  h={a['dr']:.6f} -> h={b_['dr']:.6f} : observed order p = {order:.3f}")

# =====================================================================
# 2) Point comparison table (finest grid)
# =====================================================================
# Tabulate T_num vs T_exact at a fixed set of (r/R, theta) sample points
# on the finest grid, then repeat across all grids for the CSV used in
# the convergence study.
M_fine, N_fine = grids[-1]
T_fine, info_fine = direct_results[(M_fine, N_fine)]
point_rows = compare_to_exact(T_fine, M_fine, N_fine)
with open(f'{OUT}/point_comparison_finest.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(point_rows[0].keys()))
    w.writeheader()
    w.writerows(point_rows)
print("\nPoint comparison (finest grid, direct solve):")
maxpt = 0.0
for r in point_rows:
    print(f"  r/R={r['r_frac']:.2f} theta={r['theta_deg']:>4}deg  "
          f"T_num={r['Tnum']:8.5f}  T_exact={r['Texact']:8.5f}  "
          f"abs_err={r['abs_err']:.3e}  pct_err={r['pct_err']:.5f}%")
    maxpt = max(maxpt, r['abs_err'])
print(f"Max abs error at sample points: {maxpt:.3e}")

per_grid_point_rows = []
for (M, N) in grids:
    T, info = direct_results[(M, N)]
    rows = compare_to_exact(T, M, N)
    for r in rows:
        r2 = dict(r); r2['M'] = M; r2['N'] = N
        per_grid_point_rows.append(r2)
with open(f'{OUT}/point_comparison_all_grids.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(per_grid_point_rows[0].keys()))
    w.writeheader()
    w.writerows(per_grid_point_rows)

# =====================================================================
# 3) Iterative solvers: Jacobi, GS, SOR (omega scan) -- on all grids
# =====================================================================
# For each grid level: run Jacobi and Gauss-Seidel once (omega=1), then
# scan SOR over a range of relaxation factors to find the empirically
# best omega, and record its iteration count/time too.
iter_summary = []
omega_scan_results = {}

for (M, N) in grids:
    t_start = time.time()
    Tj, infoj = iterative_solve(M, N, method='jacobi', tol=1e-6, maxiter=60000)
    iter_summary.append(dict(M=M, N=N, method='Jacobi', omega=1.0,
                              iterations=infoj['iterations'], time_s=infoj['time'],
                              mem_kb=infoj['peak_mem_kb'], converged=infoj['converged']))
    print(f"Jacobi   M={M:3d} N={N:3d}: iters={infoj['iterations']:6d} "
          f"time={infoj['time']:.3f}s converged={infoj['converged']}")

    Tg, infog = iterative_solve(M, N, method='gs', omega=1.0, tol=1e-6, maxiter=60000)
    iter_summary.append(dict(M=M, N=N, method='Gauss-Seidel', omega=1.0,
                              iterations=infog['iterations'], time_s=infog['time'],
                              mem_kb=infog['peak_mem_kb'], converged=infog['converged']))
    print(f"GS       M={M:3d} N={N:3d}: iters={infog['iterations']:6d} "
          f"time={infog['time']:.3f}s converged={infog['converged']}")

    # Coarse scan across the full range, refined with a denser sweep near
    # omega=2, since the optimal omega moves closer to 2 as the grid is
    # refined and a fixed 0.05 step would miss it on the finer grids.
    omegas = np.concatenate([np.arange(1.00, 1.80, 0.05), np.arange(1.80, 1.991, 0.01)])
    scan = []
    for om in omegas:
        om = round(float(om), 3)
        Ts, infos = iterative_solve(M, N, method='sor', omega=om, tol=1e-6, maxiter=60000)
        scan.append((om, infos['iterations'], infos['converged']))
    omega_scan_results[(M, N)] = [(float(o), int(it), bool(c)) for (o, it, c) in scan]
    scan = omega_scan_results[(M, N)]
    # Pick the empirically best (converged) omega -- fewest iterations.
    conv_scan = [s for s in scan if s[2]]
    best = min(conv_scan, key=lambda s: s[1]) if conv_scan else None
    if best:
        om_best, it_best, _ = best
        Tb_sor, infob = iterative_solve(M, N, method='sor', omega=om_best, tol=1e-6, maxiter=60000)
        iter_summary.append(dict(M=M, N=N, method='SOR (best omega)', omega=om_best,
                                  iterations=infob['iterations'], time_s=infob['time'],
                                  mem_kb=infob['peak_mem_kb'], converged=infob['converged']))
        print(f"SOR-best M={M:3d} N={N:3d}: omega={om_best:.2f} iters={it_best:6d} "
              f"time={infob['time']:.3f}s")
    print(f"  [grid {M}x{N} block done in {time.time()-t_start:.1f}s]")

with open(f'{OUT}/iterative_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(iter_summary[0].keys()))
    w.writeheader()
    w.writerows(iter_summary)

with open(f'{OUT}/omega_scan.json', 'w') as f:
    json.dump({f'{M}x{N}': scan for (M, N), scan in omega_scan_results.items()}, f, indent=2)

# =====================================================================
# 4) Energy-balance / flux check (finest grid, direct solution)
# =====================================================================
# With no internal heat generation, the surface heat flux should integrate
# to (approximately) zero around the full circumference.
integral, q = flux_integral_check(T_fine, M_fine, N_fine)
q_scale = np.max(np.abs(q))
print(f"\nFlux energy-balance check (finest grid): "
      f"integral of q''*R dtheta = {integral:.6e} W/m  "
      f"(relative to peak flux*2piR = {q_scale*2*np.pi*R:.3e})")

with open(f'{OUT}/energy_balance.txt', 'w') as f:
    f.write(f"M={M_fine}, N={N_fine}\n")
    f.write(f"Integral of q''(theta)*R dtheta over 0..2pi = {integral:.6e} W/m\n")
    f.write(f"Reference scale (max|q''| * 2*pi*R)          = {q_scale*2*np.pi*R:.6e} W/m\n")
    f.write(f"Relative magnitude                           = {abs(integral)/(q_scale*2*np.pi*R):.3e}\n")

# =====================================================================
# PLOTS
# =====================================================================
# Build a full (M+1) x (N+1) grid for plotting by duplicating the j=0
# column at j=N, so contour/pcolor routines see a closed (periodic) ring
# instead of a wedge with a seam at theta=2*pi.
dtheta = 2*np.pi/N_fine
dr = R/M_fine
theta = dtheta*np.arange(N_fine+1)
r = dr*np.arange(M_fine+1)
T_plot = np.zeros((M_fine+1, N_fine+1))
T_plot[:, :N_fine] = T_fine
T_plot[:, N_fine] = T_fine[:, 0]
RR, TT = np.meshgrid(r, theta, indexing='ij')
X = RR*np.cos(TT)              # convert polar (r,theta) grid to Cartesian for plotting
Y = RR*np.sin(TT)

# Filled isotherm contours over the cylinder cross-section (finest grid).
fig, ax = plt.subplots(figsize=(6, 6))
cs = ax.contourf(X, Y, T_plot, levels=25, cmap='inferno')
ax.contour(X, Y, T_plot, levels=25, colors='k', linewidths=0.3)
fig.colorbar(cs, ax=ax, label='T (deg C)')
ax.set_aspect('equal')
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
ax.set_title(f'Isotherms in the cylinder cross-section\n(M={M_fine}, N={N_fine}, direct solve)')
fig.tight_layout()
fig.savefig(f'{OUT}/isotherms.png', dpi=150)
plt.close(fig)

# Log-log grid-convergence plot of max error vs dr, with a slope-2
# reference line to visually confirm the expected O(h^2) accuracy.
drs = [row['dr'] for row in refinement_rows]
errs = [row['max_abs_err'] for row in refinement_rows]
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.loglog(drs, errs, 'o-', label='Observed max |error|')
ref = errs[0]*(np.array(drs)/drs[0])**2
ax.loglog(drs, ref, 'k--', label=r'Reference slope 2 ($O(h^2)$)')
ax.set_xlabel(r'$\Delta r$ (m)')
ax.set_ylabel('max |T_num - T_exact| (deg C)')
ax.set_title('Grid-convergence study (4 grid levels)')
ax.legend()
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/convergence.png', dpi=150)
plt.close(fig)

# SOR iteration count vs relaxation factor omega, one curve per grid level,
# showing the sharpening minimum near omega=2 as h shrinks.
fig, ax = plt.subplots(figsize=(6.5, 4.8))
for (M, N), scan in omega_scan_results.items():
    oms = [s[0] for s in scan if s[2]]
    its = [s[1] for s in scan if s[2]]
    ax.plot(oms, its, 'o-', label=f'M={M}, N={N}', ms=4)
ax.set_xlabel(r'relaxation factor $\omega$')
ax.set_ylabel('iterations to converge (tol=1e-6)')
ax.set_yscale('log')
ax.set_title('SOR iteration count vs relaxation factor')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/sor_omega_scan.png', dpi=150)
plt.close(fig)

# Computed surface heat flux distribution q''(theta) on the outer boundary
# (finest grid) -- should oscillate roughly symmetric about zero,
# consistent with the energy-balance check above.
fig, ax = plt.subplots(figsize=(6, 4.5))
theta_deg = np.rad2deg(dtheta*np.arange(N_fine))
q_plot = surface_flux(T_fine, M_fine, N_fine)
ax.plot(theta_deg, q_plot, 'b-', lw=1.2)
ax.axhline(0, color='k', lw=0.7)
ax.set_xlabel(r'$\theta$ (deg)')
ax.set_ylabel(r"$q''(\theta)$ (W/m$^2$)")
ax.set_title('Surface heat flux distribution (finest grid)')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/surface_flux.png', dpi=150)
plt.close(fig)

print("\nAll outputs written to", OUT)
