"""
ray tracing subroutine
from Parameter Estimation and Inverse Problems, 3rd edition, 2018
by R. Aster, B. Borchers, C. Thurber
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline


"""
[cxl,czl] = cellfunc(x1,xv,z1,zv)

find the indices of the cells that are to the upper left of the
raypath points(x1,z1)

INPUT
  x1 - a vector of data points along the x axis
  xv - a vector of the bins along the x axis
  z1 - a vector of data points along the z axis
  zv - a vector of the bins along the z axis

OUTPUT
  cxl - a vector of the index of the greatest element in xv that is less than 
        x1
  czl - a vector of the index of the greatest element in zv that is less than 
        z1
"""



def cellfunc(x1, xv, z1, zv):
    xv = np.asarray(xv)
    zv = np.asarray(zv)
    x1 = np.asarray(x1)
    z1 = np.asarray(z1)

    # clamp inside grid
    x1c = np.clip(x1, xv[0], xv[-1])
    z1c = np.clip(z1, zv[0], zv[-1])

    cxl = np.searchsorted(xv, x1c, side='right') - 1
    czl = np.searchsorted(zv, z1c, side='right') - 1

    cxl = np.clip(cxl, 0, len(xv) - 2)
    czl = np.clip(czl, 0, len(zv) - 2)
    return cxl.astype(int), czl.astype(int)


"""
vmid=vel2(xmid,zmid,ip,kp,xn,zn,v);

2-d velocity interpolation function. Bilinear interpolation of velocity at (xmid, zmid) using cell upper-left indices ip,kp (zero-based).

INPUT
  xmid - the x midpoint of the ray segment
  zmid - the z midpoint of the ray segment
  ip   - the first index to the cell of v containing xmid, zmid
  kp   - the second index to the cell of v containing xmid, zmid
  xn   - the x cell divisions
  zn   - the z cell divisions
  v    - the velocity structure
"""


def vel2(xmid, zmid, ip, kp, xn, zn, v):
    xmid = np.atleast_1d(xmid)
    zmid = np.atleast_1d(zmid)
    ip = np.asarray(ip).astype(int)
    kp = np.asarray(kp).astype(int)
    xn = np.asarray(xn)
    zn = np.asarray(zn)
    v = np.asarray(v)

    x0 = xn[ip]
    x1 = xn[ip + 1]
    z0 = zn[kp]
    z1 = zn[kp + 1]

    dx = np.where((x1 - x0) != 0, (x1 - x0), 1.0)
    dz = np.where((z1 - z0) != 0, (z1 - z0), 1.0)

    tx = np.clip((xmid - x0) / dx, 0.0, 1.0)
    tz = np.clip((zmid - z0) / dz, 0.0, 1.0)

    v00 = v[ip,     kp    ]
    v10 = v[ip + 1, kp    ]
    v01 = v[ip,     kp + 1]
    v11 = v[ip + 1, kp + 1]

    v0 = (1 - tx) * v00 + tx * v10
    v1 = (1 - tx) * v01 + tx * v11
    return (1 - tz) * v0 + tz * v1

"""
[dvx,dvz]=vel2d(x2,z2,ip,kp,xn,zn,v);

2-d derivative interpolation function. Gradients (dv/dx, dv/dz) of the bilinear interpolant at (x2,z2). Vectorized.

INPUT
  x2 - the x midpoint of the ray segment
  z2 - the z midpoint of the ray segment
  ip - the first index to the cell of v containing xmid, zmid
  kp - the second index to the cell of v containing xmid, zmid
  xn - the x cell divisions
  zn - the z cell divisions
  v  - the velocity structure

OUTPUT
  dvx - an approximation of the derivative of the velocity model in the x 
        direction
  dvz - an approximation of the derivative of the velocity model in the z 
        direction
"""


def vel2d(x2, z2, ip, kp, xn, zn, v):
    x2 = np.atleast_1d(x2)
    z2 = np.atleast_1d(z2)
    ip = np.asarray(ip).astype(int)
    kp = np.asarray(kp).astype(int)
    xn = np.asarray(xn)
    zn = np.asarray(zn)
    v = np.asarray(v)

    x0 = xn[ip];      x1 = xn[ip + 1]
    z0 = zn[kp];      z1 = zn[kp + 1]

    dx = np.where((x1 - x0) != 0, (x1 - x0), 1.0)
    dz = np.where((z1 - z0) != 0, (z1 - z0), 1.0)

    tx = np.clip((x2 - x0) / dx, 0.0, 1.0)
    tz = np.clip((z2 - z0) / dz, 0.0, 1.0)

    v00 = v[ip,     kp    ]
    v10 = v[ip + 1, kp    ]
    v01 = v[ip,     kp + 1]
    v11 = v[ip + 1, kp + 1]

    # dv/dx
    dv_dx = ((1 - tz) * (v10 - v00) / dx) + (tz * (v11 - v01) / dx)
    # dv/dz
    dv_dz = ((1 - tx) * (v01 - v00) / dz) + (tx * (v11 - v10) / dz)
    return dv_dx, dv_dz


"""
[vmid,wv]=vel2w(xmid,zmid,ip,kp,xn,zn,v);

Generates a vector of raypath mid points velocities and the necessary
functions (wv(:,i)) to calculate the Jacobian. Bilinear interpolation returning both the velocity and the 4 bilinear weightsnneeded for the Jacobian accumulation.

INPUT
  xmid - the x midpoint of the ray segment
  zmid - the z midpoint of the ray segment
  ip   - the first index to the cell of v containing xmid, zmid
  kp   - the second index to the cell of v containing xmid, zmid
  xn   - the x cell divisions
  zn   - the z cell divisions
  v    - the velocity structure

OUTPUT
  vmid - the interpolated velocity for the points
  wv   - the factors needed to compute the Jacobian
"""


def vel2w(xmid, zmid, ip, kp, xn, zn, v):
    xmid = np.atleast_1d(xmid)
    zmid = np.atleast_1d(zmid)
    ip = np.asarray(ip).astype(int)
    kp = np.asarray(kp).astype(int)
    xn = np.asarray(xn)
    zn = np.asarray(zn)
    v = np.asarray(v)

    x0 = xn[ip];      x1 = xn[ip + 1]
    z0 = zn[kp];      z1 = zn[kp + 1]

    dx = np.where((x1 - x0) != 0, (x1 - x0), 1.0)
    dz = np.where((z1 - z0) != 0, (z1 - z0), 1.0)

    tx = np.clip((xmid - x0) / dx, 0.0, 1.0)
    tz = np.clip((zmid - z0) / dz, 0.0, 1.0)

    w00 = (1 - tx) * (1 - tz)
    w10 = tx * (1 - tz)
    w01 = (1 - tx) * tz
    w11 = tx * tz

    v00 = v[ip,     kp    ]
    v10 = v[ip + 1, kp    ]
    v01 = v[ip,     kp + 1]
    v11 = v[ip + 1, kp + 1]

    vmid = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11
    wv = np.stack([w00, w10, w01, w11], axis=1)
    return vmid, wv

"""
    FORWARD PROBLEM: Ray tracing + travel time calculation only
    
    Given a velocity model, compute travel times for all source-receiver pairs
    using bent-ray tracing (pseudo-bending method).
    
    Parameters
    ----------
    PSCALE : int
        Grid dimension (sources, receivers, velocity nodes)
    NIT : int
        Maximum ray bending iterations
    CONV : float
        Convergence tolerance for ray path
    XFAC : float
        Convergence enhancement factor
    xn, zn : ndarray
        Node positions
    v : ndarray (PSCALE, PSCALE)
        Velocity model
    sc : ndarray (PSCALE, 2)
        Source coordinates
    rc : ndarray (PSCALE, 2)
        Receiver coordinates
    calcrays : int, optional
        0 = initialize rays, 1 = use rpinit
    rpinit : ndarray, optional
        Initial ray paths from previous iteration
    
    Returns
    -------
    ttcal : ndarray (PSCALE, PSCALE)
        Calculated travel times for all source-receiver pairs
    rpstore : ndarray (PSCALE^2, NSEG+1, 2)
        Ray path coordinates for all source-receiver pairs
    
    Notes
    -----
    This is the "forward model" that predicts data (travel times) 
    from model parameters (velocity field).
    """


def forward_problem(PSCALE, NIT, CONV, XFAC, xn, zn, v, sc, rc, calcrays=0, rpinit=None):
    
    NSEG = PSCALE * 2
    ndata = PSCALE * PSCALE
    
    ttcal = np.zeros((PSCALE, PSCALE))
    rpstore = np.zeros((ndata, NSEG + 1, 2))
    
    rpnum = 0
    for j in range(PSCALE):
        xs, zs = sc[j, 0], sc[j, 1]
        for k in range(PSCALE):
            xr, zr = rc[k, 0], rc[k, 1]
            rpnum += 1
            
            dx0 = (xr - xs) / NSEG
            dz0 = (zr - zs) / NSEG
            
            # straight-line initial path
            xp = np.linspace(xs, xr, NSEG + 1)
            zp = zs + np.arange(NSEG + 1) * dz0
            if calcrays == 0:
                rp = np.column_stack([xp, zp])
            else:
                rp = rpinit[rpnum - 1, :, :].copy()
            
            # ==========================================
            # RAY BENDING (Fermat's principle)
            # ==========================================
            ivec = np.arange(1, NSEG)  # interior points
            for _ in range(NIT):
                rpnew = rp.copy()
                
                # midpoints of neighbors
                x2a = 0.5 * (rp[ivec + 1, 0] + rp[ivec - 1, 0])
                z2a = 0.5 * (rp[ivec + 1, 1] + rp[ivec - 1, 1])
                
                dxa = rp[ivec + 1, 0] - rp[ivec - 1, 0]
                dza = rp[ivec + 1, 1] - rp[ivec - 1, 1]
                dna = dxa * dxa + dza * dza
                ddna = np.sqrt(np.maximum(dna, 1e-30))
                rdx = dxa / ddna
                rdz = dza / ddna
                
                # velocity info at x2a, z2a
                cxul, czul = cellfunc(x2a, xn, z2a, zn)
                vmid = vel2(x2a, z2a, cxul, czul, xn, zn, v)
                vx, vz = vel2d(x2a, z2a, cxul, czul, xn, zn, v)
                
                vrd = vx * rdx + vz * rdz
                rvx = vx - vrd * rdx
                rvz = vz - vrd * rdz
                rvs = np.sqrt(rvx * rvx + rvz * rvz)
                
                # update interior points using pseudo-bending
                for i_loc in range(len(ivec)):
                    ii = ivec[i_loc]
                    xxk = x2a[i_loc]
                    zzk = z2a[i_loc]
                    if rvs[i_loc] != 0.0:
                        rvx_i = rvx[i_loc] / rvs[i_loc]
                        rvz_i = rvz[i_loc] / rvs[i_loc]
                        rcur = vmid[i_loc] / rvs[i_loc]
                        inside = max(rcur * rcur - 0.25 * dna[i_loc], 0.0)
                        rtemp = rcur - np.sqrt(inside)
                        xxk = x2a[i_loc] + XFAC * rvx_i * rtemp
                        zzk = z2a[i_loc] + XFAC * rvz_i * rtemp
                    rpnew[ii, 0] = xxk
                    rpnew[ii, 1] = zzk
                
                # check convergence
                diff = rp - rpnew
                if np.linalg.norm(diff.ravel()) / max(np.linalg.norm(rp.ravel()), 1e-30) < CONV:
                    rp = rpnew
                    break
                rp = rpnew
            
            rpstore[rpnum - 1, :, :] = rp
            
            # ==========================================
            # TRAVEL TIME CALCULATION
            # ==========================================
            jvec = np.arange(NSEG)
            xmida = 0.5 * (rp[jvec + 1, 0] + rp[jvec, 0])
            zmida = 0.5 * (rp[jvec + 1, 1] + rp[jvec, 1])
            dxa = rp[jvec + 1, 0] - rp[jvec, 0]
            dza = rp[jvec + 1, 1] - rp[jvec, 1]
            ra = np.sqrt(dxa * dxa + dza * dza)  # segment lengths
            
            cxul, czul = cellfunc(xmida, xn, zmida, zn)
            vmid2w, _ = vel2w(xmida, zmida, cxul, czul, xn, zn, v)
            
            # travel time = ∫ (1/v) dl = Σ (segment_length / velocity)
            ttcal[j, k] = np.sum(ra / vmid2w)
    
    return ttcal, rpstore

"""
    JACOBIAN COMPUTATION: Sensitivity matrix (∂t/∂m)
    
    Compute the Jacobian matrix that relates changes in model parameters
    (slowness = 1/velocity) to changes in data (travel times).
    
    Parameters
    ----------
    PSCALE : int
        Grid dimension
    xn, zn : ndarray
        Node positions
    v : ndarray (PSCALE, PSCALE)
        Velocity model
    sc : ndarray (PSCALE, 2)
        Source coordinates
    rc : ndarray (PSCALE, 2)
        Receiver coordinates
    rpstore : ndarray (PSCALE^2, NSEG+1, 2)
        Ray paths for all source-receiver pairs (from forward_problem)
    
    Returns
    -------
    J : ndarray (PSCALE^2, PSCALE^2)
        Jacobian matrix where J[i,j] = ∂t_i/∂m_j
        i = observation index (source-receiver pair)
        j = model parameter index (slowness at grid node)
    
    Notes
    -----
    The Jacobian for travel time tomography is straightforward:
        ∂t/∂m_j = length of ray segment passing through cell j
    
    We use bilinear interpolation weights to distribute the sensitivity
    to the 4 nodes surrounding each ray segment midpoint.
    
    Physical meaning:
        J[i,j] tells you: "If I increase slowness at node j by 1,
        how much does travel time i increase?"
    """


def compute_jacobian(PSCALE, xn, zn, v, sc, rc, rpstore):
    NSEG = PSCALE * 2
    nmodel = PSCALE * PSCALE
    ndata = PSCALE * PSCALE
    
    J = np.zeros((ndata, nmodel))
    
    # Fortran-style linear index (match MATLAB)
    def idxF(i, j):  # zero-based i,j
        return i + j * PSCALE
    
    rpnum = 0
    for j in range(PSCALE):
        for k in range(PSCALE):
            rpnum += 1
            
            # get the ray path for this source-receiver pair
            rp = rpstore[rpnum - 1, :, :]
            
            # ==========================================
            # JACOBIAN ACCUMULATION
            # ==========================================
            # For each segment of this ray:
            # 1. Find which 4 nodes surround the segment midpoint
            # 2. Compute bilinear interpolation weights
            # 3. Add (segment_length × weight) to J[observation, node]
            
            jvec = np.arange(NSEG)
            xmida = 0.5 * (rp[jvec + 1, 0] + rp[jvec, 0])
            zmida = 0.5 * (rp[jvec + 1, 1] + rp[jvec, 1])
            dxa = rp[jvec + 1, 0] - rp[jvec, 0]
            dza = rp[jvec + 1, 1] - rp[jvec, 1]
            ra = np.sqrt(dxa * dxa + dza * dza)  # segment lengths
            
            cxul, czul = cellfunc(xmida, xn, zmida, zn)
            vmid2w, wv = vel2w(xmida, zmida, cxul, czul, xn, zn, v)
            
            # row index in J (Fortran ordering like MATLAB)
            nobs = j + PSCALE * k
            
            # accumulate Jacobian row
            for i_seg in range(NSEG):
                i0 = cxul[i_seg]
                j0 = czul[i_seg]
                
                # four nodes around the segment midpoint
                node1 = idxF(i0,     j0    )  # lower-left
                node2 = idxF(i0 + 1, j0    )  # lower-right
                node3 = idxF(i0,     j0 + 1)  # upper-left
                node4 = idxF(i0 + 1, j0 + 1)  # upper-right
                
                # bilinear weights (sum to 1)
                w00, w10, w01, w11 = wv[i_seg, :]
                
                # J[obs, node] = ray_segment_length × interpolation_weight
                # Physical meaning: sensitivity of travel time to slowness
                J[nobs, node1] += ra[i_seg] * w00
                J[nobs, node2] += ra[i_seg] * w10
                J[nobs, node3] += ra[i_seg] * w01
                J[nobs, node4] += ra[i_seg] * w11
    
    return J

"""
    Gauss-Newton iterations to find velocity model for a given regularization parameter
    
    Parameters
    ----------
    alpha : float
        Regularization parameter (will be scaled by ||J||_2)
    PSCALE : int
        Grid dimension
    NIT : int
        Ray bending iterations
    CONV : float
        Ray convergence tolerance
    XFAC : float
        Ray bending enhancement factor
    MAXITER : int
        Number of Gauss-Newton iterations
    xn, zn : ndarray
        Velocity model node positions
    xni, zni : ndarray
        Interpolation grid for plotting
    sc : ndarray (PSCALE, 2)
        Source coordinates
    rc : ndarray (PSCALE, 2)
        Receiver coordinates
    ttobs : ndarray (PSCALE, PSCALE)
        Observed travel times
    vbackground : float
        Background velocity for initial guess
    L : ndarray (n, n)
        Roughening matrix (discrete Laplacian)
    mtrue : ndarray, optional
        True model for computing error (if available)
    
    Returns
    -------
    v_final : ndarray (PSCALE, PSCALE)
        Final velocity model
    vi_final : ndarray
        Interpolated velocity for plotting
    misfit : ndarray (MAXITER,)
        Residual norm at each iteration
    mnorm : ndarray (MAXITER,)
        Model seminorm at each iteration
    mrms : ndarray (MAXITER,)
        Model error at each iteration (if mtrue provided)
    rparam : float
        Actual regularization parameter used (alpha * ||J||_2)
    """






    
def gauss_newton(m0, Cm_inv, Cd_inv, PSCALE, NIT, CONV, XFAC, MAXITER,
                 xn, zn, xni, zni, sc, rc, ttobs, vbackground, mtrue):
    
  

    # Initialize model: constant background velocity
    v    = vbackground * np.ones((PSCALE, PSCALE))
    m    = (1.0 / v).reshape(-1, order='F')

    # Ray tracing state
    calcrays = 0
    rpinit   = None

    # Diagnostics
    misfit = np.zeros(MAXITER)
    mrms   = np.zeros(MAXITER)

    for iter_idx in range(1, MAXITER + 1):

        # ── FORWARD PROBLEM AND JACOBIAN ─────────────────────────────────────
        if iter_idx == 1:
            ttcal, rpstore = forward_problem(PSCALE, NIT, CONV, XFAC,
                                             xn, zn, v, sc, rc, calcrays, rpinit)
            J = compute_jacobian(PSCALE, xn, zn, v, sc, rc, rpstore)
        else:
            J     = K
            ttcal = tttry

        calcrays = 1
        rpinit   = rpstore

        # ── RESIDUAL ─────────────────────────────────────────────────────────
        rvec = (ttcal - ttobs).reshape(-1, order='F')

        # ── GAUSS-NEWTON SOLVE ───────────────────────────────────────────────
        # System: (J' Cd_inv J + Cm_inv) dm = -(J'r + Cm_inv @ (m - m0))
        J1  = J.T @ Cd_inv @ J + Cm_inv
        rhs = -(J.T @ Cd_inv @ rvec + Cm_inv @ (m - m0))
        dm  = np.linalg.solve(J1, rhs)

        # ── UPDATE MODEL ─────────────────────────────────────────────────────
        m    = m + dm
        v    = 1.0 / m.reshape((PSCALE, PSCALE), order='F')

        # ── FORWARD MODEL FOR NEXT ITERATION ─────────────────────────────────
        tttry, rpstore = forward_problem(PSCALE, NIT, CONV, XFAC,
                                         xn, zn, v, sc, rc, calcrays, rpinit)
        K = compute_jacobian(PSCALE, xn, zn, v, sc, rc, rpstore)

        # ── DIAGNOSTICS ──────────────────────────────────────────────────────
        misfit[iter_idx - 1] = np.linalg.norm((ttobs - tttry).reshape(-1, order='F'))

        if mtrue is not None:
            mrms[iter_idx - 1] = np.linalg.norm(mtrue - m)
        else:
            mrms[iter_idx - 1] = np.nan

    # ── INTERPOLATE FINAL MODEL FOR PLOTTING ─────────────────────────────────
    spline_final = RectBivariateSpline(xn, zn, v)
    vi           = spline_final(xni, zni)

    return v, vi, misfit, mrms





"""
ttstor=plotraypaths(PSCALE,NIT,CONV,XFAC,xn,zn,v,sc,rc);

INPUT
  PSCALE - the length of a side of v
  NIT    - the maximum number of iterations to perform determing the path
  CONV   - the relative change in travel time that is acceptable for 
           convergence
  XFAC   - a factor used to control convergance
  xn     - the x positions of the nodes in v
  zn     - the z positions of the nodes in v
  v      - the seismic velocity grid (PSCALE by PSCALE matrix)
  sc     - the coordinates of the seismic sources
  rc     - the coordinates of the seismic receivers

OUTPUT
  ttstor - the travel times between each source and each receive

ttstor is the matrix of travel times corresponding to the ray tracing
for the source points sc, receiver points rc, and velocity model v
indexed at xn and zn.  Size of the problem is PSCALE x PSCALE

This function also adds the ray paths to the current plot.
"""



def plotraypaths(PSCALE, NIT, CONV, XFAC, xn, zn, v, sc, rc):
    """
    Bent-ray pseudo-bending (Aster/Borchers/Thurber Ex. 10.1) + plotting.
    Returns:
      ttstor : (PSCALE x PSCALE) travel times between each source and receiver.
    Behavior mirrors the MATLAB function you shared, including convergence test.
    """
    ttstor = np.zeros((PSCALE, PSCALE))
    nseg = PSCALE * 2  # number of segments per ray

    for j in range(PSCALE):
        xs, zs = sc[j, 0], sc[j, 1]
        for k in range(PSCALE):
            xr, zr = rc[k, 0], rc[k, 1]

            # evenly split difference in each direction
            dx = (xr - xs) / nseg
            dz = (zr - zs) / nseg

            # initial straight-line path (nseg+1 points)
            xp = np.linspace(xs, xr, nseg + 1)
            zp = zs + np.arange(nseg + 1) * dz
            rp = np.column_stack([xp, zp])  # shape (nseg+1, 2)

            # initial travel-time estimate
            tt = 0.0
            seg_len = np.sqrt(dx * dx + dz * dz)

            for i in range(nseg):
                xmid = 0.5 * (rp[i+1, 0] + rp[i, 0])
                zmid = 0.5 * (rp[i+1, 1] + rp[i, 1])
                cxul, czul = cellfunc(xmid, xn, zmid, zn)
                vmid = vel2(xmid, zmid, cxul, czul, xn, zn, v)
                tt += seg_len / vmid

            ttlast = tt

            # bending convergence loop
            for _ in range(NIT):
                tt = 0.0
                rpnew = rp.copy()

                # update interior points
                for i in range(1, nseg):
                    x2 = 0.5 * (rp[i+1, 0] + rp[i-1, 0])
                    z2 = 0.5 * (rp[i+1, 1] + rp[i-1, 1])
                    xxk, zzk = x2, z2

                    cxul, czul = cellfunc(x2, xn, z2, zn)
                    vmid = vel2(x2, z2, cxul, czul, xn, zn, v)
                    vx, vz = vel2d(x2, z2, cxul, czul, xn, zn, v)

                    dxp = rp[i+1, 0] - rp[i-1, 0]
                    dzp = rp[i+1, 1] - rp[i-1, 1]
                    dn = dxp * dxp + dzp * dzp
                    ddn = np.sqrt(dn) if dn > 0 else 1.0
                    rdx = dxp / ddn
                    rdz = dzp / ddn

                    vrd = vx * rdx + vz * rdz
                    rvx = vx - vrd * rdx
                    rvz = vz - vrd * rdz
                    rvs = np.sqrt(rvx * rvx + rvz * rvz)

                    if rvs != 0.0:
                        rvx /= rvs
                        rvz /= rvs
                        rcur = vmid / rvs
                        inside = rcur * rcur - 0.25 * dn
                        # Guard against tiny negatives due to floating error
                        inside = max(inside, 0.0)
                        rtemp = rcur - np.sqrt(inside)

                        # convergence enhancement
                        xxk = x2 + XFAC * rvx * rtemp
                        zzk = z2 + XFAC * rvz * rtemp

                    rpnew[i, 0] = xxk
                    rpnew[i, 1] = zzk

                # accept new path
                rp = rpnew

                # recompute travel time
                tt = 0.0
                for i in range(1, nseg + 1):
                    xmid = 0.5 * (rp[i, 0] + rp[i-1, 0])
                    zmid = 0.5 * (rp[i, 1] + rp[i-1, 1])
                    cxul, czul = cellfunc(xmid, xn, zmid, zn)
                    vmid = vel2(xmid, zmid, cxul, czul, xn, zn, v)
                    dxs = rp[i, 0] - rp[i-1, 0]
                    dzs = rp[i, 1] - rp[i-1, 1]
                    tt += np.sqrt(dxs * dxs + dzs * dzs) / vmid

                # convergence check
                if ttlast != 0 and abs(ttlast - tt) / abs(ttlast) < CONV:
                    break
                ttlast = tt

            # store travel time
            ttstor[j, k] = tt

            # plot the ray
            plt.plot(rp[:, 0], rp[:, 1], 'k--', linewidth=0.8)

    # match MATLAB axis style used in the function
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(min(xn), max(xn))
    ax.set_ylim(min(zn), max(zn))
#     ax.set_xlabel('m')
#     ax.set_ylabel('m')
    # Note: MATLAB used axis(..., 'ij') which flips vertical; we keep standard y-up
    return ttstor