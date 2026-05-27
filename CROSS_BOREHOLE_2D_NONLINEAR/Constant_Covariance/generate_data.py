
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.interpolate import RectBivariateSpline
from scipy.io import loadmat


from functions import *
np.random.seed(42)

import os


n_samples = 50000

print(f"n_samples = {n_samples}")

# Create directory if it doesn't exist
DATA = f"DATA"
os.makedirs(DATA, exist_ok=True)

# -------------------------- SETUP PARAMETERS ----------------------------------------

NIT = 50               # ray-bending iterations
SCALE = 1600           # meters
PSCALE = 7             # receivers, sources, sqrt(num velocity nodes)
XFAC = 1.2             # convergence enhancement factor
MAXITER = 10            # Gauss–Newton linearization steps
CONV = 1e-4            # travel-time convergence
NOISE = 0.001          # seconds

# Geometry: sources, receivers
scz = np.linspace(0, SCALE, PSCALE)
scx = np.zeros_like(scz)
sc = np.column_stack([scx, scz])

rcz = np.linspace(0, SCALE, PSCALE)
rcx = SCALE * np.ones_like(rcz)
rc = np.column_stack([rcx, rcz])

# Velocity model node positions
xn = np.linspace(-150, SCALE + 150, PSCALE)
zn = xn.copy()

# Centers of fast/slow anomalies
xfast = xn[3 * PSCALE // 4 - 1]
zfast = zn[3 * PSCALE // 4 - 1]
xslow = xn[PSCALE // 2 - 1]
zslow = zn[PSCALE // 2 - 1]

print(f"  Grid size: {PSCALE}×{PSCALE}")
print(f"  Model domain: {SCALE}m × {SCALE}m")
print(f"  Noise level: {NOISE} seconds")


# -------------------------- TRUE MODEL ----------------------------------------

# True velocity model
vtrue = np.zeros((PSCALE, PSCALE))
vbackground = 2900.0

for i in range(PSCALE):
    for j in range(PSCALE):
        term_fast = 0.10 * np.exp(-0.00004 * ((xn[i] - xfast) ** 2 + (zn[j] - zfast) ** 2))
        term_slow = 0.15 * np.exp(-0.00004 * ((xn[i] - xslow) ** 2 + (zn[j] - zslow) ** 2))
        vtrue[i, j] = vbackground * (1 + term_fast) * (1 - term_slow)

# True slowness model (parameters)
mtrue = (1.0 / vtrue).reshape(-1, order='F')


# -------------------------- SIMULATED DATA ----------------------------------------

def synthetic_data(PSCALE, NIT, CONV, XFAC, xn, zn, vtrue, sc, rc, seed=0):
    rng = np.random.default_rng(seed)
    epsilon = rng.normal(0, NOISE, (PSCALE, PSCALE))
    ttcal_clean, _ = forward_problem(PSCALE, NIT, CONV, XFAC, xn, zn, vtrue, sc, rc)
    ttobs = ttcal_clean + epsilon
    return ttobs


ttobs = synthetic_data(PSCALE, NIT, CONV, XFAC, xn, zn, vtrue, sc, rc, seed=0)

# Save true model and observations
np.savetxt(os.path.join(DATA,"m_true.csv"), mtrue, delimiter=",")
np.savetxt(os.path.join(DATA,"d_obs.csv"), ttobs, delimiter=",")


print(f"Shape: {ttobs.shape}")
print(f"  True velocity range: {np.min(vtrue):.1f} to {np.max(vtrue):.1f} m/s")
print(f"  Travel time range: {np.min(ttobs):.4f} to {np.max(ttobs):.4f} seconds")


# Smooth plotting grid via interpolation (like interp2 cubic)
# ----------------------------
xnm, znm = np.meshgrid(xn, zn, indexing='ij')  # model grid

# Interpolation grid
xni = np.arange(-150, SCALE + 150 + 1, 10)     # finer grid
zni = np.arange(-150, SCALE + 150 + 1, 10)
xnim, znim = np.meshgrid(xni, zni, indexing='ij')

# Use a spline for smooth interpolation akin to 'cubic'
spline = RectBivariateSpline(xn, zn, vtrue)   # note: expects ascending 1D coords
vi = spline(xni, zni)                         # shape (len(xni), len(zni))


plt.figure(1)
plt.clf()
im = plt.imshow(
    vi.T,
    extent=[xni.min(), xni.max(), zni.min(), zni.max()],
    origin='lower',
    aspect='equal',
    cmap='viridis',
    vmin=2600, vmax=3200,
    interpolation='nearest'
)
plt.colorbar(im)
tstor = plotraypaths(PSCALE, NIT, CONV, XFAC, xn, zn, vtrue, sc, rc)
ax = plt.gca()
ax.set_ylim(zni.max(), zni.min())   
ax.set_xlim(xni.min(), xni.max())
plt.xlabel('x(m)'); plt.ylabel('z(m)')
plt.title('True model and ray paths')
plt.savefig("true_model_raypaths.png", dpi=300, bbox_inches='tight')
plt.show()

# -------------------------- PRIOR MEAN AND COVARIANCE ----------------------------------------

n = PSCALE**2
m0    = (1.0 / vbackground) * np.ones(n)

sigma_m = (0.05 * (1.0 / vbackground))

Cm = sigma_m**2 * np.eye(n)
Cm_inv  = np.linalg.inv(Cm)



# check balance
v0 = vbackground * np.ones((PSCALE, PSCALE))
ttcal0, rp0 = forward_problem(PSCALE, NIT, CONV, XFAC, xn, zn, v0, sc, rc)
J0 = compute_jacobian(PSCALE, xn, zn, v0, sc, rc, rp0)


Cd_inv  = (1.0 / NOISE**2) * np.eye(PSCALE * PSCALE)


print(f"||J^T J||_2   = {np.linalg.norm(J0.T @ J0, 2):.3e}")
print(f"||Cm_inv||_2  = {np.linalg.norm(Cm_inv, 2):.3e}")


# -------------------------- GAUSS NEWTON ----------------------------------------


v_GN, vi_GN, misfit, mrms = gauss_newton(m0, Cm_inv, Cd_inv, PSCALE, NIT, CONV, 
                                                        XFAC, MAXITER, xn, zn, xni, zni, 
                                                        sc, rc, ttobs, vbackground, mtrue)

m_GN = (1.0 / v_GN).reshape(-1, order='F')

np.savetxt(os.path.join(DATA,"m_GN.csv"), m_GN, delimiter=",")

# ------------------------------------------------
# plotting slowness mtrue and m_GN
# ------------------------------------------------
mtrue_2d = mtrue.reshape((PSCALE, PSCALE), order="F")
m_GN_2d = m_GN.reshape((PSCALE, PSCALE), order="F")


fig, axes = plt.subplots(1, 2, figsize=(12,5), sharey=True)

# ---- True slowness ----
im1 = axes[0].imshow(
    mtrue_2d.T,
    extent=[xn.min(), xn.max(), zn.min(), zn.max()],
    origin='lower',
    aspect='equal',
    cmap='viridis',
    interpolation='nearest'
)

axes[0].set_title("True slowness")
axes[0].set_xlabel("x (m)")
axes[0].set_ylabel("z (m)")
axes[0].set_ylim(zn.max(), zn.min())
axes[0].set_xlim(xn.min(), xn.max())

# ---- Sample mean ----
im2 = axes[1].imshow(
    m_GN_2d.T,
    extent=[xn.min(), xn.max(), zn.min(), zn.max()],
    origin='lower',
    aspect='equal',
    cmap='viridis',
    interpolation='nearest'
)

axes[1].set_title("Gauss-Newton")
axes[1].set_xlabel("x (m)")
axes[1].set_ylabel("z (m)")
axes[1].set_ylim(zn.max(), zn.min())
axes[1].set_xlim(xn.min(), xn.max())

# ---- shared colorbar ----
cbar = fig.colorbar(im2, ax=axes, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label("Slowness")

plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# ---- True model + rays ----
im1 = axes[0].imshow(vi.T,
    extent=[xni.min(), xni.max(), zni.min(), zni.max()],
    origin='lower',
    aspect='equal',
    cmap='viridis',
   
    interpolation='nearest'
)
axes[0].set_title('True model and ray paths')
axes[0].set_xlabel('x (m)')
axes[0].set_ylabel('z (m)')
axes[0].set_ylim(zni.max(), zni.min())   # re-enforce limits
axes[0].set_xlim(xni.min(), xni.max())




# ---- Gauss-Newton model ----
im0 = axes[1].imshow(vi_GN.T,
    extent=[xni.min(), xni.max(), zni.min(), zni.max()],
    origin='lower',
    aspect='equal',
    cmap='viridis',
  
    interpolation='nearest'
)
axes[1].set_title('Gauss-Newton model')
axes[1].set_xlabel('x (m)')
axes[1].set_ylabel('z (m)')
axes[1].set_ylim(zni.max(), zni.min())   # MATLAB 'ij'
axes[1].set_xlim(xni.min(), xni.max())


# ---- Single shared colorbar ----
cbar = fig.colorbar(im0, ax=axes, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('Velocity (m/s)')

# plt.tight_layout()
plt.show()


# -------------------------- SAMPLING ----------------------------------------

def generate_observations_2d(m_samples, PSCALE, NIT, CONV, XFAC,
                              xn, zn, sc, rc, NOISE, seed=11):
    """
    For each m_i in m_samples, compute d_i = G(m_i) + eps_i.
    Returns d with shape (N, PSCALE*PSCALE).
    """
    rng = np.random.default_rng(seed)
    N   = m_samples.shape[0]
    n_d = PSCALE * PSCALE

    d_noiseless = np.zeros((N, n_d))
    for i in range(N):
        v_i = (1.0 / m_samples[i]).reshape(PSCALE, PSCALE, order='F')
        tt_clean, _ = forward_problem(PSCALE, NIT, CONV, XFAC,
                                      xn, zn, v_i, sc, rc)
        d_noiseless[i] = tt_clean.reshape(-1, order='F')
        if i % 100 == 0:
            print(f"  {i}/{N}", end="\r")

    eps = rng.normal(0, NOISE, size=(N, n_d))
    d   = d_noiseless + eps
    return d

# -------------------------- GAUSSIAN PRIOR ----------------------------------------

def sample_gaussian_prior_2d(n_samples, m0, Cm, seed=10):
    """
    Draw n_samples from the Gaussian prior m ~ N(m0, Cm).
    Returns m_gauss with shape (n_samples, n).
    """
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean=m0, cov=Cm, size=n_samples)

m_gauss = sample_gaussian_prior_2d(n_samples, m0, Cm, seed=10)

d_gauss = generate_observations_2d(m_gauss, PSCALE, NIT, CONV, XFAC,
                                    xn, zn, sc, rc, NOISE, seed=11)

np.savetxt(os.path.join(DATA, "m_gaussian.csv"), m_gauss, delimiter=",")
np.savetxt(os.path.join(DATA, "d_gaussian.csv"), d_gauss, delimiter=",")

# -------------------------- LAPLACE PRIOR ----------------------------------------

def sample_laplace_prior_2d(n_samples, m0, Cm, seed=20):
    rng = np.random.default_rng(seed)
    n = m0.size
    b = np.sqrt(np.diag(Cm) / 2.0)              # shape (n,)
    eta = rng.laplace(0.0, 1.0, size=(n_samples, n)) * b[np.newaxis, :]
    m = m0.reshape(1, -1) + eta
    return m
m_laplace = sample_laplace_prior_2d(n_samples, m0, Cm, seed=20)

d_laplace = generate_observations_2d(m_laplace, PSCALE, NIT, CONV, XFAC,
                                      xn, zn, sc, rc, NOISE, seed=21)

np.savetxt(os.path.join(DATA, "m_laplace.csv"), m_laplace, delimiter=",")
np.savetxt(os.path.join(DATA, "d_laplace.csv"), d_laplace, delimiter=",")

# -------------------------- TV PRIOR ----------------------------------------
def sample_tv_2d(n_samples, m0, sigma_m, PSCALE, seed=30):
    
    rng    = np.random.default_rng(seed)
    samp   = np.zeros((n_samples, PSCALE * PSCALE))
    b_grad = sigma_m / np.sqrt(2 * PSCALE)   # calibrated to domain size

    for i in range(n_samples):
        # Independent x-direction walk: each column gets its own walk
        ux = np.zeros((PSCALE, PSCALE))
        for ix in range(1, PSCALE):
            ux[ix, :] = ux[ix-1, :] + rng.laplace(0.0, b_grad, size=PSCALE)

        # Independent z-direction walk: each row gets its own walk
        uz = np.zeros((PSCALE, PSCALE))
        for iz in range(1, PSCALE):
            uz[:, iz] = uz[:, iz-1] + rng.laplace(0.0, b_grad, size=PSCALE)

        # Sum and mean-centre
        u      = ux + uz
        u     -= np.mean(u)
        samp[i] = m0 + u.reshape(-1, order='F')

    # Rescale using mean marginal std (seed/N-independent)
    perturb           = samp - m0.reshape(1, -1)
    mean_marginal_std = perturb.std(axis=0).mean()
    perturb          *= sigma_m / mean_marginal_std
    return m0.reshape(1, -1) + perturb

m_tv = sample_tv_2d(n_samples, m0, sigma_m, PSCALE, seed=30)
d_tv = generate_observations_2d(m_tv, PSCALE, NIT, CONV, XFAC,
                                      xn, zn, sc, rc, NOISE, seed=31)

np.savetxt(os.path.join(DATA, "m_tv.csv"), m_tv, delimiter=",")
np.savetxt(os.path.join(DATA, "d_tv.csv"), d_tv, delimiter=",")

# -------------------------- UNIFORM PRIOR ----------------------------------------

def sample_uniform_prior_2d(n_samples, m0, sigma_m, seed=40):
    
    rng       = np.random.default_rng(seed)
    n         = m0.size
    a_uniform = sigma_m * np.sqrt(3)
    eta       = rng.uniform(-a_uniform, a_uniform, size=(n_samples, n))
    return m0.reshape(1, -1) + eta
m_uniform = sample_uniform_prior_2d(n_samples, m0, sigma_m, seed=40)
d_uniform = generate_observations_2d(m_uniform, PSCALE, NIT, CONV, XFAC,
                                      xn, zn, sc, rc, NOISE, seed=41)

np.savetxt(os.path.join(DATA, "m_uniform.csv"), m_uniform, delimiter=",")
np.savetxt(os.path.join(DATA, "d_uniform.csv"), d_uniform, delimiter=",")


# --------------------------SANITY CHECK ----------------------------------------
datasets = [
    ("Gaussian (L2)",   m_gauss,   d_gauss,   "#2196F3"),
    ("Laplace (L1)",    m_laplace, d_laplace, "#FF9800"),
    ("Total Variation", m_tv,      d_tv,      "#4CAF50"),
    ("Uniform",         m_uniform, d_uniform, "#9C27B0"),
]

fig, axes = plt.subplots(4, 2, figsize=(14, 18))

for row, (label, ms, ds, color) in enumerate(datasets):

    # ── sample mean of m (2D image) ──────────────────────────────────────
    m_mean   = ms.mean(axis=0).reshape((PSCALE, PSCALE), order='F')
    mtrue_2d = mtrue.reshape((PSCALE, PSCALE), order='F')

    ax = axes[row, 0]
    im = ax.imshow(m_mean.T,
                   extent=[xn.min(), xn.max(), zn.min(), zn.max()],
                   origin='lower', aspect='equal',
                   cmap='viridis', interpolation='nearest')
    # overlay mtrue as contour for comparison
    ax.contour(xn, zn, mtrue_2d.T, colors='white', linewidths=1.0, alpha=0.7)
    fig.colorbar(im, ax=ax)
    ax.set_title(f"{label} — sample mean slowness", fontsize=10)
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    ax.set_ylim(zn.max(), zn.min())

    # ── sample mean of d (observation heatmap) ───────────────────────────
    d_mean   = ds.mean(axis=0).reshape((PSCALE, PSCALE), order='F')
    ttobs_2d = ttobs.reshape((PSCALE, PSCALE), order='F') \
               if ttobs.shape == (PSCALE, PSCALE) else ttobs

    ax = axes[row, 1]
    im2 = ax.imshow(d_mean.T,
                    origin='lower', aspect='equal',
                    cmap='viridis', interpolation='nearest')
    # overlay observed travel times as white dots
    ax.contour(d_mean.T, colors='white', linewidths=0.8, alpha=0.6)
    fig.colorbar(im2, ax=ax)
    ax.set_title(f"{label} — sample mean travel times", fontsize=10)
    ax.set_xlabel("source index"); ax.set_ylabel("receiver index")

plt.suptitle("Training Data Sanity Check\n"
             r"(white contours = $m_{\rm true}$ / $d_{\rm obs}$)",
             fontsize=13)
plt.tight_layout()
plt.savefig("sanity_check_2d.png", dpi=150)
plt.show()




# ── print statistics ─────────────────────────────────────────────────────
print("\n" + "="*52)
print(f" {'Dataset':<22} {'m std':>8} {'d std':>8}")
print("="*52)
for label, ms, ds, _ in datasets:
    print(f" {label:<22} {ms.std():>8.3e} {ds.std():>8.3e}")
print("="*52)