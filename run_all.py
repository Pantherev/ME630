"""
Problem 6 -- driver script.

Calls the functions defined in solver.py to produce: Task (a) numbers,
the grid-refinement study for the direct solve, the point-by-point
comparison table, the Jacobi/Gauss-Seidel/SOR comparison, and the three
plots (temperature profile, convergence, SOR omega scan). All results
are also written out to CSV/JSON files.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')          # use a non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import csv, json, os

from solver import (r0, r1, r2, k1, k2, Ti, Tinf, h, analytical_resistances,
                     T_exact, build_grid, solve_direct, iterative_solve,
                     compare_to_exact, max_error)

OUT = 'out'          # folder where all tables/plots are saved
os.makedirs(OUT, exist_ok=True)

# =============================================================================
# Task (a): thermal-resistance network calculation
# -----------------------------------------------------------------------
# Evaluate the closed-form formulas once and print/save them as the
# "by hand" benchmark.
# =============================================================================
res = analytical_resistances()
print("Task (a) -- thermal-resistance network:")
for k_, v in res.items():
    print(f"  {k_} = {v:.6f}")

# Consistency check: does the convective boundary condition at r2,
# evaluated with the computed T(r2), reproduce the same heat rate Q'
# that we started from?
Q_check = h*(res['T_r2'] - Tinf)*2*np.pi*r2
print(f"  Check: h*(T(r2)-Tinf)*2*pi*r2 = {Q_check:.6f} W/m "
      f"(should equal Q' = {res['Qp']:.6f} W/m)")

with open(f'{OUT}/task_a_resistances.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['quantity', 'value', 'units'])
    w.writerow(["R1' (metal wall)", f"{res['R1p']:.6e}", 'm.K/W'])
    w.writerow(["R2' (insulation)", f"{res['R2p']:.6e}", 'm.K/W'])
    w.writerow(["R3' (convection)", f"{res['R3p']:.6e}", 'm.K/W'])
    w.writerow(["Q' (heat rate/length)", f"{res['Qp']:.6f}", 'W/m'])
    w.writerow(["T(r1) (interface)", f"{res['T_r1']:.6f}", 'deg C'])
    w.writerow(["T(r2) (outer surface)", f"{res['T_r2']:.6f}", 'deg C'])
    w.writerow(["Check: h(T(r2)-Tinf)(2 pi r2)", f"{Q_check:.6f}", 'W/m'])

# =============================================================================
# Task (c): grid refinement -- direct (Thomas) solve
# -----------------------------------------------------------------------
# Solve the same physical problem on three different grid spacings
# (2.5, 1.0, 0.5 mm) using the direct Thomas-algorithm solver, and track
# how the maximum error shrinks as the grid is refined.
# =============================================================================
drs_mm = [2.5, 1.0, 0.5]
grids = [dr_mm/1000.0 for dr_mm in drs_mm]   # convert mm to metres

refinement_rows = []
direct_results = {}   # keep every solution around so later sections can reuse it

for dr in grids:
    T, r, n1, M, info = solve_direct(dr)
    direct_results[dr] = (T, r, n1, M, info)
    maxerr = max_error(T, r)
    refinement_rows.append(dict(dr_mm=dr*1000, N=info['N'], time_ms=info['time']*1e3,
                                 mem_kb=info['peak_mem_kb'], max_abs_err=maxerr))
    print(f"Direct(Thomas) dr={dr*1000:.1f}mm: N={info['N']:4d} "
          f"time={info['time']*1e3:.4f} ms mem={info['peak_mem_kb']:.2f} kB "
          f"max|err|={maxerr:.3e}")

with open(f'{OUT}/refinement_direct.csv', 'w', newline='') as f:
    wtr = csv.DictWriter(f, fieldnames=list(refinement_rows[0].keys()))
    wtr.writeheader(); wtr.writerows(refinement_rows)

# Observed order of accuracy between successive grids: p = ln(e1/e2)/ln(dr1/dr2).
# For a proper second-order scheme this should come out close to 2.
print("\nObserved order of accuracy (Thomas solve, max abs error over whole domain):")
for a, b_ in zip(refinement_rows[:-1], refinement_rows[1:]):
    order = np.log(a['max_abs_err']/b_['max_abs_err']) / np.log(a['dr_mm']/b_['dr_mm'])
    print(f"  dr={a['dr_mm']}mm -> dr={b_['dr_mm']}mm : observed order p = {order:.3f}")

# =============================================================================
# Point comparison at r0, r1, (r1+r2)/2, r2 -- for every grid
# -----------------------------------------------------------------------
# Compares the numerical solution against the analytical one AT SPECIFIC
# INTERIOR POINTS, with both absolute and percentage error reported.
# =============================================================================
point_rows = []
r_mid = (r1 + r2)/2.0

for dr in grids:
    T, r, n1, M, info = direct_results[dr]
    for label, r_target in [('r0', r0), ('r1', r1), ('(r1+r2)/2', r_mid), ('r2', r2)]:
        # Find the grid index closest to the point we want, and make sure
        # it really does land exactly on a grid node (it should, since the
        # points chosen all coincide with r0, r1 or r2, and (r1+r2)/2 works
        # out to a grid node for the chosen dr values).
        idx = int(round((r_target - r0)/dr))
        assert abs(r[idx] - r_target) < 1e-9, f"{label} not on grid for dr={dr}"
        Tn = T[idx]
        Te = T_exact(r[idx])
        err = abs(Tn - Te)
        pct = 100*err/abs(Te)
        point_rows.append(dict(dr_mm=dr*1000, point=label, r=r[idx], Tnum=Tn,
                                Texact=Te, abs_err=err, pct_err=pct))
        print(f"  dr={dr*1000:>4.1f}mm {label:>10s} (r={r[idx]:.4f}): "
              f"T_num={Tn:9.5f} T_exact={Te:9.5f} abs_err={err:.3e} pct_err={pct:.5f}%")

with open(f'{OUT}/point_comparison.csv', 'w', newline='') as f:
    wtr = csv.DictWriter(f, fieldnames=list(point_rows[0].keys()))
    wtr.writeheader(); wtr.writerows(point_rows)

# =============================================================================
# Iterative solvers: Jacobi, Gauss-Seidel, SOR (omega scan) -- all 3 grids
# -----------------------------------------------------------------------
# Solves the same system iteratively too, and records how many sweeps
# each method needs to converge. For SOR the relaxation factor omega is
# also scanned to find the fastest-converging value.
# =============================================================================
iter_summary = []
omega_scan_results = {}

for dr in grids:
    # --- Jacobi ---
    Tj, r, n1, M, infoj = iterative_solve(dr, method='jacobi', tol=1e-6, maxiter=2_000_000)
    iter_summary.append(dict(dr_mm=dr*1000, method='Jacobi', omega=1.0,
                              iterations=infoj['iterations'], time_s=infoj['time'],
                              mem_kb=infoj['peak_mem_kb'], converged=infoj['converged']))
    print(f"Jacobi dr={dr*1000:>4.1f}mm: iters={infoj['iterations']:6d} "
          f"time={infoj['time']:.4f}s converged={infoj['converged']}")

    # --- Gauss-Seidel (SOR with omega = 1) ---
    Tg, r, n1, M, infog = iterative_solve(dr, method='gs', omega=1.0, tol=1e-6, maxiter=2_000_000)
    iter_summary.append(dict(dr_mm=dr*1000, method='Gauss-Seidel', omega=1.0,
                              iterations=infog['iterations'], time_s=infog['time'],
                              mem_kb=infog['peak_mem_kb'], converged=infog['converged']))
    print(f"GS dr={dr*1000:>4.1f}mm: iters={infog['iterations']:6d} "
          f"time={infog['time']:.4f}s converged={infog['converged']}")

    # --- SOR: scan omega from 1.0 to just under 2.0 in steps of 0.05, and
    # keep the number of iterations needed at each value, so we can plot
    # "iterations vs omega" and read off the empirically-best omega. ---
    omegas = np.arange(1.0, 1.99, 0.05)
    scan = []
    for om in omegas:
        Ts, r, n1, M, infos = iterative_solve(dr, method='sor', omega=om, tol=1e-6, maxiter=2_000_000)
        scan.append((float(om), int(infos['iterations']), bool(infos['converged'])))
    omega_scan_results[dr] = scan

    # Pick the omega that converged in the fewest iterations, and record
    # that as "SOR (best omega)" in the summary table.
    conv_scan = [s for s in scan if s[2]]
    best = min(conv_scan, key=lambda s: s[1]) if conv_scan else None
    if best:
        om_best, it_best, _ = best
        Tb_, r, n1, M, infob = iterative_solve(dr, method='sor', omega=om_best, tol=1e-6, maxiter=2_000_000)
        iter_summary.append(dict(dr_mm=dr*1000, method='SOR (best omega)', omega=om_best,
                                  iterations=infob['iterations'], time_s=infob['time'],
                                  mem_kb=infob['peak_mem_kb'], converged=infob['converged']))
        print(f"SOR-best dr={dr*1000:>4.1f}mm: omega={om_best:.2f} iters={it_best:6d} "
              f"time={infob['time']:.4f}s")

with open(f'{OUT}/iterative_summary.csv', 'w', newline='') as f:
    wtr = csv.DictWriter(f, fieldnames=list(iter_summary[0].keys()))
    wtr.writeheader(); wtr.writerows(iter_summary)

# =============================================================================
# PLOTS
# =============================================================================

# --- (a) Temperature profile: numerical (finest grid) vs analytical -------
# Shows how nearly all the temperature drop happens across the insulation
# layer, while the steel wall stays almost isothermal (because R2' >> R1').
dr_fine = grids[-1]
T_fine, r_fine, n1_fine, M_fine, info_fine = direct_results[dr_fine]
r_dense = np.linspace(r0, r2, 400)          # a fine radius array just for a smooth analytical curve
T_dense_exact = T_exact(r_dense)

fig, ax = plt.subplots(figsize=(6.5, 4.8))
ax.plot(r_dense*1000, T_dense_exact, 'k-', lw=1.6, label='Analytical (thermal-resistance)')
ax.plot(r_fine*1000, T_fine, 'ro', ms=3, label=f'Numerical (Thomas, dr={dr_fine*1000:.1f} mm)')
ax.axvline(r1*1000, color='gray', ls='--', lw=0.8)   # mark the material interface
ax.text(r1*1000+0.5, ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),
        'metal | insulation\ninterface', fontsize=8, color='gray')
ax.set_xlabel('r (mm)'); ax.set_ylabel('T (deg C)')
ax.set_title('Problem 6: temperature distribution through composite shell')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/temperature_profile.png', dpi=150)
plt.close(fig)

# --- (b) grid convergence: max abs error vs dr (log-log) ------------------
# Plotting error against grid spacing on a log-log scale lets us confirm
# the O(h^2) accuracy visually -- a slope of 2 on this plot means
# second-order convergence.
drs_arr = [row['dr_mm'] for row in refinement_rows]
errs = [row['max_abs_err'] for row in refinement_rows]

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.loglog(drs_arr, errs, 'o-', label='Observed max |error|')
# Reference line with a slope of exactly 2, anchored to the first data
# point, so we can compare our observed curve against ideal 2nd-order behaviour
ref = errs[0]*(np.array(drs_arr)/drs_arr[0])**2
ax.loglog(drs_arr, ref, 'k--', label=r'Reference slope 2 ($O(h^2)$)')
ax.set_xlabel(r'$\Delta r$ (mm)')
ax.set_ylabel('max |T_num - T_exact| (deg C)')
ax.set_title('Problem 6: grid-convergence study')
ax.legend()
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/convergence.png', dpi=150)
plt.close(fig)

# --- (c) SOR omega scan -----------------------------------------------------
# Shows how many iterations SOR needs for each relaxation factor omega,
# for every grid -- lets us see where the optimum omega sits and how it
# shifts as the grid is refined.
fig, ax = plt.subplots(figsize=(6, 4.5))
for dr, scan in omega_scan_results.items():
    oms = [s[0] for s in scan if s[2]]   # only keep omegas that actually converged
    its = [s[1] for s in scan if s[2]]
    ax.plot(oms, its, 'o-', label=f'dr={dr*1000:.1f} mm')
ax.set_xlabel(r'relaxation factor $\omega$')
ax.set_ylabel('iterations to converge (tol=1e-6)')
ax.set_yscale('log')
ax.set_title('Problem 6: SOR iteration count vs relaxation factor')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/sor_omega_scan.png', dpi=150)
plt.close(fig)

# Also dump the raw omega-scan numbers to JSON, in case they're needed for
# double-checking the plot later.
with open(f'{OUT}/omega_scan.json', 'w') as f:
    json.dump({f'{dr*1000:.1f}mm': scan for dr, scan in omega_scan_results.items()}, f, indent=2)

print("\nAll outputs written to", OUT)
