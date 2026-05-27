import numpy as np

# Jacobian
def Gfun(x, z, w,H):
    """
    x : (n,) array
    z : (m,) array  (z(w))
    w : (m,) array
    returns G of shape (n, m)
    """
    x = np.asarray(x).ravel()   # (n,)
    z = np.asarray(z).ravel()   # (m,)
    w = np.asarray(w).ravel()   # (m,)

    X = x[:, None]   # (n,1)
    W = w[None, :]   # (1,m)
    Z = z[None, :]   # (1,m) -> broadcast over rows

    G = 2.0 * (H - Z) / (((X - W) ** 2) + (H - Z) ** 2)
    return G

# # g = g(x,w,z(w))
# def forward_map(x, w, z,H):
#     """
#     x   : (n,)array
#     w   : (m,) array
#     z   : (m,) array  (z(w))
#     returns g(w) of shape (n,)
#     """
#     w = np.asarray(w).ravel()
#     z = np.asarray(z).ravel()
#     x   = np.asarray(x).ravel()
    
#     n = x.size
#     m = w.size
    
#     if z.size != m:
#         raise ValueError(f"z must have length m={m}, got {z.size}")
    

#     g = np.empty(n, dtype=float)
    
#     for j in range(n):
#         X = float(x[j])
#         num = (X - w) ** 2 + H ** 2
#         den = (X - w) ** 2 + (H - z) ** 2
#         ffun = np.log(num / den) # should be (m,) 
        
#         f = np.asarray(ffun).ravel()
        
#         g[j] = np.trapz(f, w)
            
#     return g

#Newton's method
def inv_DDCP(x, w, H, d, sigma_d, z0, Cz, K,forward_map):
    """
    Python/Numpy port of inv_DDCP.m

    Parameters
    ----------
    w : (m,) array_like, Integration grid for the continuous parameter (Tarantola's w).
        
    x : (n,) array_like, Data locations (surface parameter).
        
    d : (n,) array_like, Observed data vector.
    sigma_d : float, Standard uncertainty of the data (added on S diagonal as in the MATLAB code).
        
    sigma_z : float, Prior standard deviation used in the covariance kernel.
        
    Cz : (m,m) matrix_like, Prior Covariance
    
    Delta : float
        Length-scale parameter used in the covariance kernel denominator.
    K : int
        Number of iterations.
    Gfun : callable
        Function Jfun(x, z, w,H) -> Jacobian matrix of shape (n, m).
        
    forward_map : callable
        Function forward_map (x_j, w, z) -> integral over w returning shape (n,m).

    Returns
    -------
    zhat : (m, K) ndarray
        Columns are the successive iterates z_{k+1}(w).
    """
    w   = np.asarray(w).ravel()
    x   = np.asarray(x).ravel()
    d   = np.asarray(d).ravel()

    m = w.size
    n = x.size

    # --- Assertions (mirroring the MATLAB checks) ---
    assert n != 1, "Data is not a column vector (expected length != 1)."
    assert m != 1, "Need more than one data set (m must be > 1)."
    assert sigma_d > 0, "sigma_d must be positive."
    assert Cz.shape[0] == Cz.shape[1], "Covariance matrix must be square."
    assert np.all(np.diag(Cz) > 0), "Covariance matrix has zero or negative values on the diagonal!"
    

    # --- Covariance matrix: C_p(w,w') = sigma_z^2 * exp( -0.5 * (w-w')^2 / Delta ) ---
    cov = Cz  # (m,m)

    # Initialize
    z = z0
    zhat = np.zeros((m, K), dtype=float)

    # Helper: trapezoidal integral along w given array shaped (..., m) if needed
    # Here we’ll be explicit with axis where we integrate along the w-dimension.
    for k in range(K):
        # Jacobian at current iterate
        G = Gfun(x, z, w, H)  # expected shape (n, m)
        if G.shape != (n, m):
            raise ValueError(f"Gfun must return shape (n, m) = {(n, m)}, got {G.shape}")

        # ----- Build S (n x n) -----
        # S(i,j) = ∫_w  [ G(i,:)^T  ∘  ( ∫_{w'} cov(w, w') ∘ G(j, w') dw' ) ]  dw
        # where ∘ denotes pointwise multiplication and integrals are trapz over w.
        S = np.empty((n, n), dtype=float)

        # Precompute H_j(w) = ∫_{w'} cov(w, w') * G(j, w') dw'  for each j.
        # cov (m,m), G(j,:) -> (m,), broadcast to (m,m) as row; integrate along w' (axis=1).
        H_cols = []
        for j in range(n):
            Hj = np.trapz(cov * G[j, :][None, :], w, axis=1)  # (m,)
            H_cols.append(Hj)
        H_cols = np.stack(H_cols, axis=1)  # (m, n), column j is H_j

        # Now S(i,j) = ∫_w G(i, w) * H_j(w) dw
        for i in range(n):
            # Vector of S(i, :) by integrating G(i,:) against each H_j
            # Do a single trapz over w for every j by multiplying G(i,:)[:,None] * H
            S[i, :] = np.trapz(G[i, :][:, None] * H_cols, w, axis=0)

        # Diagonal adjustment: S(i,i) += sigma_d  (matches the MATLAB code)
        S[np.diag_indices_from(S)] += sigma_d

        # ----- Build part2 (length n) -----
        # I5(j) = ∫_w G(j,w) * z(w) dw
        I5 = np.trapz(G * z[None, :], w, axis=1)  # (n,)

        # g(j) = ∫_w ffun(x_j, w, z) dw
        g = forward_map(x, w, z,H)

        part2 = d - g + I5  # (n,)

        # ----- Compute z_{k+1}(w) -----
        # MATLAB: all = sum_{i,j} cov .* G(i,:) * Sinv(i,j) * part2(j)
        # Let u = S^{-1} * part2  -> then all = sum_i cov .* G(i,:) * u_i
        # where cov .* G(i,:) multiplies each column j of cov by G(i,j).
        # Vectorized approach:
        # Solve S u = part2 (avoid explicit inverse)
        u = np.linalg.solve(S, part2)  # (n,)

        # Accumulate "all" as an (m,m) matrix
        all_mat = np.zeros((m, m), dtype=float)
        for i in range(n):
            # Broadcast: cov (m,m) * G(i,:)[None,:] scales each column j by G(i,j)
            all_mat += cov * G[i, :][None, :] * u[i]

        # Integrate along columns (over w') to get length-m vector
        I6 = np.trapz(all_mat, w, axis=1)  # (m,)
        z = I6
        zhat[:, k] = I6

    return zhat


