# ── Imports ───────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from functions import forward_problem, compute_jacobian

# ── Reproducibility ───────────────────────────────────────────────────
SEED = 42
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

# ── Directories ───────────────────────────────────────────────────────
DATA_DIR    = "DATA
RESULTS_DIR = "RESULTS_PINN"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Problem geometry ──────────────────────────────────────────────────
NIT         = 50
XFAC        = 1.2
CONV        = 1e-4
SCALE       = 1600
PSCALE      = 7
vbackground = 2900.0
NOISE       = 0.001
n_param     = PSCALE * PSCALE    # 49
n_obs       = PSCALE * PSCALE    # 49

xn = np.linspace(-150, SCALE + 150, PSCALE)
zn = xn.copy()

scz = np.linspace(0, SCALE, PSCALE)
scx = np.zeros_like(scz)
sc  = np.column_stack([scx, scz])
rcz = np.linspace(0, SCALE, PSCALE)
rcx = SCALE * np.ones_like(rcz)
rc  = np.column_stack([rcx, rcz])

m0 = (1.0 / vbackground) * np.ones(n_param)   # ≈ 3.448e-04 s/m

# ── Precompute linearised forward operator ────────────────────────────
print("Precomputing Jacobian J0 at background model ...")
v0       = vbackground * np.ones((PSCALE, PSCALE))
_, rp0   = forward_problem(PSCALE, NIT, CONV, XFAC, xn, zn, v0, sc, rc)
J0_np    = compute_jacobian(PSCALE, xn, zn, v0, sc, rc, rp0)   # (49, 49)
d_bg_np  = (J0_np @ m0).astype(np.float32)                     # (49,)
J0_tf    = tf.constant(J0_np,   dtype=tf.float32)
d_bg_tf  = tf.constant(d_bg_np, dtype=tf.float32)
m0_tf    = tf.constant(m0.astype(np.float32))
print(f"  ||J0||_2 = {np.linalg.norm(J0_np, 2):.3e}")

# ── Load fixed reference data ─────────────────────────────────────────
m_true = np.loadtxt(os.path.join(DATA_DIR, "m_true.csv"), delimiter=",")
d_obs  = np.loadtxt(os.path.join(DATA_DIR, "d_obs.csv"),  delimiter=",")
m_GN   = np.loadtxt(os.path.join(DATA_DIR, "m_GN.csv"),   delimiter=",")

d_obs = d_obs.reshape(-1, order='F')
m_GN  = m_GN.reshape(-1, order='F') if m_GN.ndim == 2 else m_GN

# ── Prior file map ────────────────────────────────────────────────────
PRIORS = {
    "Gaussian": ("m_gaussian.csv", "d_gaussian.csv"),
    "Laplace":  ("m_laplace.csv",  "d_laplace.csv"),
    "TV":       ("m_tv.csv",       "d_tv.csv"),
    "Uniform":  ("m_uniform.csv",  "d_uniform.csv"),
}

# ── Hyperparameters ───────────────────────────────────────────────────
epochs         = 1000
batch_size     = 256
lr             = 1e-3
l2_strength    = 1e-4
dropout        = 0.2
Patience       = 40
wd             = 1e-4
clipnorm       = 1.0
lambda_phys    = 1.0
plateau_patience = 15

hidden_units  = (128, 256, 512, 256, 128)
cnn2d_filters = [32, 64, 128]
cnn2d_kernel  = (3, 3)
cnn2d_dense   = [256, 128]
p       = 128
width   = 256
depth   = 4
n_freqs = 16


# ══════════════════════════════════════════════════════════════════════
# Physics-informed loss
# ══════════════════════════════════════════════════════════════════════
def make_physics_loss(m_mean_tf, m_std_tf):
    def physics_informed_loss(y_batch_scaled, m_pred_scaled, d_orig_batch):
        mse_m       = tf.reduce_mean((y_batch_scaled - m_pred_scaled) ** 2)
        m_pred_phys = m_pred_scaled * m_std_tf + m_mean_tf   # (B, 49)
        dm_pred     = m_pred_phys - m0_tf                    # (B, 49)
        d_pred      = tf.linalg.matmul(dm_pred, J0_tf,
                                        transpose_b=True) + d_bg_tf  # (B,49)
        phys  = tf.reduce_mean((d_pred - d_orig_batch) ** 2)
        total = mse_m + lambda_phys * phys
        return total, mse_m, phys
    return physics_informed_loss


# ══════════════════════════════════════════════════════════════════════
# Train / val step factories
# model_is_2d flag controls whether X_batch is (B,49) or (B,7,7,1).
# d_orig is always recovered from X_flat (B,49) regardless of shape.
# ══════════════════════════════════════════════════════════════════════
def make_train_step(model, optimizer, loss_fn, model_is_2d=False):
    @tf.function
    def train_step(X_batch_flat, X_batch_model, y_batch, d_orig_batch):
        with tf.GradientTape() as tape:
            m_pred            = model(X_batch_model, training=True)
            loss, mse_m, phys = loss_fn(y_batch, m_pred, d_orig_batch)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, mse_m, phys
    return train_step


def make_val_step(model, loss_fn):
    @tf.function
    def val_step(X_batch_flat, X_batch_model, y_batch, d_orig_batch):
        m_pred            = model(X_batch_model, training=False)
        loss, mse_m, phys = loss_fn(y_batch, m_pred, d_orig_batch)
        return loss, mse_m, phys
    return val_step


# ══════════════════════════════════════════════════════════════════════
# Custom training loop
# ══════════════════════════════════════════════════════════════════════
def train_physics_informed(model, X_train_flat, X_train_model,
                           X_val_flat, X_val_model,
                           y_train, y_val,
                           d_orig_train_np, d_orig_val_np,
                           scaler_m_mean_np, scaler_m_scale_np):

    optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=lr, weight_decay=wd, clipnorm=clipnorm
    )

    m_mean_tf = tf.constant(scaler_m_mean_np,  dtype=tf.float32)
    m_std_tf  = tf.constant(scaler_m_scale_np, dtype=tf.float32)

    loss_fn    = make_physics_loss(m_mean_tf, m_std_tf)
    train_step = make_train_step(model, optimizer, loss_fn)
    val_step   = make_val_step(model, loss_fn)

    # Convert to tf.constant once
    X_tr_flat  = tf.constant(X_train_flat,    dtype=tf.float32)
    X_tr_model = tf.constant(X_train_model,   dtype=tf.float32)
    y_tr       = tf.constant(y_train,         dtype=tf.float32)
    d_tr       = tf.constant(d_orig_train_np, dtype=tf.float32)

    X_v_flat   = tf.constant(X_val_flat,      dtype=tf.float32)
    X_v_model  = tf.constant(X_val_model,     dtype=tf.float32)
    y_v        = tf.constant(y_val,           dtype=tf.float32)
    d_v        = tf.constant(d_orig_val_np,   dtype=tf.float32)

    N     = X_tr_flat.shape[0]
    steps = int(np.ceil(N / batch_size))

    best_val         = np.inf
    best_weights     = model.get_weights()
    patience_counter = 0
    plateau_counter  = 0
    current_lr       = lr

    history = {
        "loss": [], "val_loss": [],
        "mse_m": [], "phys_loss": [],
        "val_mse_m": [], "val_phys_loss": [],
    }

    for epoch in range(epochs):
        # Shuffle all data streams in lockstep
        idx        = tf.random.shuffle(tf.range(N))
        X_sh_flat  = tf.gather(X_tr_flat,  idx)
        X_sh_model = tf.gather(X_tr_model, idx)
        y_sh       = tf.gather(y_tr,       idx)
        d_sh       = tf.gather(d_tr,       idx)

        ep_loss, ep_mse, ep_phys = [], [], []

        for step in range(steps):
            sl = slice(step * batch_size, (step + 1) * batch_size)
            l, ms, ph = train_step(
                X_sh_flat[sl],  X_sh_model[sl],
                y_sh[sl],       d_sh[sl],
            )
            ep_loss.append(float(l))
            ep_mse.append(float(ms))
            ep_phys.append(float(ph))

        val_loss, val_mse, val_phys = val_step(
            X_v_flat, X_v_model, y_v, d_v
        )
        val_loss = float(val_loss)
        val_mse  = float(val_mse)
        val_phys = float(val_phys)

        history["loss"].append(float(np.mean(ep_loss)))
        history["val_loss"].append(val_loss)
        history["mse_m"].append(float(np.mean(ep_mse)))
        history["phys_loss"].append(float(np.mean(ep_phys)))
        history["val_mse_m"].append(val_mse)
        history["val_phys_loss"].append(val_phys)

        if val_loss < best_val:
            best_val         = val_loss
            best_weights     = model.get_weights()
            patience_counter = 0
            plateau_counter  = 0
        else:
            patience_counter += 1
            plateau_counter  += 1

        if plateau_counter >= plateau_patience:
            current_lr = max(current_lr * 0.5, 1e-6)
            optimizer.learning_rate.assign(current_lr)
            plateau_counter = 0

        if patience_counter >= Patience:
            print(f"    Early stop at epoch {epoch + 1}, "
                  f"best val_loss={best_val:.6f}")
            break

    model.set_weights(best_weights)
    return history


# ══════════════════════════════════════════════════════════════════════
# Model builders — identical to NN_2D_all_priors.py
# ══════════════════════════════════════════════════════════════════════
def build_mlp_model(input_dim, output_dim):
    inputs = keras.Input(shape=(input_dim,), name="d_flat")
    x  = inputs
    e1 = None
    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, activation=None,
                          kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("gelu")(x)
        x = layers.Dropout(dropout)(x)
        if i == 0:
            e1 = x
    residual = layers.Dense(hidden_units[-1], use_bias=False,
                             activation=None)(e1)
    x = layers.Add(name="res_add")([x, residual])
    outputs = layers.Dense(output_dim, activation="linear", name="m")(x)
    return keras.Model(inputs, outputs, name="MLP")


def build_cnn2d_model(output_dim):
    inp = keras.Input(shape=(PSCALE, PSCALE, 1), name="d_2d")
    x   = inp
    for n_filt in cnn2d_filters:
        x = layers.Conv2D(n_filt, cnn2d_kernel, padding="same",
                           activation=None,
                           kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("gelu")(x)
        x = layers.Conv2D(n_filt, cnn2d_kernel, padding="same",
                           activation=None,
                           kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("gelu")(x)
        x = layers.Dropout(dropout)(x)
    x = layers.Flatten()(x)
    for units in cnn2d_dense:
        x = layers.Dense(units, activation=None,
                          kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("gelu")(x)
        x = layers.Dropout(dropout)(x)
    out = layers.Dense(output_dim, activation="linear", name="m")(x)
    return keras.Model(inp, out, name="CNN2D")


def build_deeponet_mlp_fourier(branch_dim, n_z):
    x_norm = (xn - xn.min()) / (xn.max() - xn.min())
    z_norm = (zn - zn.min()) / (zn.max() - zn.min())
    idx    = np.arange(n_z)
    px     = x_norm[idx % PSCALE]
    pz     = z_norm[idx // PSCALE]
    cols   = [px.reshape(-1, 1), pz.reshape(-1, 1)]
    for k in range(1, n_freqs + 1):
        cols.append(np.sin(2.0 * np.pi * k * px).reshape(-1, 1))
        cols.append(np.cos(2.0 * np.pi * k * px).reshape(-1, 1))
        cols.append(np.sin(2.0 * np.pi * k * pz).reshape(-1, 1))
        cols.append(np.cos(2.0 * np.pi * k * pz).reshape(-1, 1))
    phi       = np.concatenate(cols, axis=1).astype(np.float32)
    phi_const = tf.constant(phi, dtype=tf.float32)

    phi_dim   = 2 + 4 * n_freqs
    trunk_inp = keras.Input(shape=(phi_dim,), name="trunk_input")
    t_x = trunk_inp
    for _ in range(depth):
        t_x = layers.Dense(width, activation=None,
                             kernel_regularizer=keras.regularizers.l2(l2_strength))(t_x)
        t_x = layers.BatchNormalization()(t_x)
        t_x = layers.Activation("gelu")(t_x)
        t_x = layers.Dropout(dropout)(t_x)
    t_x = layers.Dense(p, activation="linear",
                        kernel_regularizer=keras.regularizers.l2(l2_strength),
                        name="trunk_out")(t_x)
    trunk_model = keras.Model(trunk_inp, t_x, name="trunk_fourier_2d")

    branch_inp = keras.Input(shape=(branch_dim,), name="branch_input")
    b = branch_inp
    for _ in range(depth):
        b = layers.Dense(width, activation=None,
                          kernel_regularizer=keras.regularizers.l2(l2_strength))(b)
        b = layers.BatchNormalization()(b)
        b = layers.Activation("gelu")(b)
        b = layers.Dropout(dropout)(b)
    b = layers.Dense(p, activation="linear",
                      kernel_regularizer=keras.regularizers.l2(l2_strength),
                      name="branch_out")(b)
    T = layers.Lambda(
        lambda b_dummy: trunk_model(phi_const, training=False),
        name="trunk_eval")(b)
    out = layers.Lambda(
        lambda inputs: tf.einsum("bp,jp->bj", inputs[0], inputs[1]),
        name="m_linear")([b, T])
    return keras.Model(branch_inp, out, name="DeepONet_MLP_Fourier")


def compile_model(model):
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr, weight_decay=wd, clipnorm=clipnorm),
        loss="mse", metrics=["mae"],
    )
    return model

def rel_l2(m_pred, m_ref):
    return float(np.linalg.norm(m_pred - m_ref) / np.linalg.norm(m_ref))

def rmse_fn(m_pred, m_ref):
    return float(np.sqrt(np.mean((m_pred - m_ref) ** 2)))


# ══════════════════════════════════════════════════════════════════════
# Main training loop — all four priors
# ══════════════════════════════════════════════════════════════════════
all_predictions = {}
all_metrics     = {}
all_histories   = {}

for prior_name, (m_file, d_file) in PRIORS.items():
    print(f"\n{'='*60}")
    print(f"  Prior: {prior_name}  [PINN]")
    print(f"{'='*60}")

    M  = np.loadtxt(os.path.join(DATA_DIR, m_file), delimiter=",")
    Gz = np.loadtxt(os.path.join(DATA_DIR, d_file), delimiter=",")

    if M.ndim  == 3: M  = M.reshape(M.shape[0],  -1, order='F')
    if Gz.ndim == 3: Gz = Gz.reshape(Gz.shape[0], -1, order='F')

    m_mean_prior = np.mean(M, axis=0)

    scaler_G  = StandardScaler()
    Gz_scaled = scaler_G.fit_transform(Gz)           # (N, 49)

    scaler_m  = StandardScaler()
    M_scaled  = scaler_m.fit_transform(M)            # (N, 49)

    scaler_m_mean_np  = scaler_m.mean_.astype(np.float32)
    scaler_m_scale_np = scaler_m.scale_.astype(np.float32)

    # Test observation — flat and 2D versions
    d_scaled_flat = scaler_G.transform(d_obs.reshape(1, -1))  # (1, 49)
    d_scaled_2d   = d_scaled_flat.reshape(1, PSCALE, PSCALE, 1)

    X_train_flat, X_val_flat, y_train, y_val = train_test_split(
        Gz_scaled, M_scaled, test_size=0.15, random_state=42,
    )

    # 2D shaped versions for CNN2D
    X_train_2d = X_train_flat.reshape(-1, PSCALE, PSCALE, 1)
    X_val_2d   = X_val_flat.reshape(-1,   PSCALE, PSCALE, 1)

    # d_orig — unscaled physical travel times, from flat scaled X
    d_orig_train = (X_train_flat * scaler_G.scale_ +
                    scaler_G.mean_).astype(np.float32)
    d_orig_val   = (X_val_flat   * scaler_G.scale_ +
                    scaler_G.mean_).astype(np.float32)

    input_dim  = X_train_flat.shape[1]   # 49
    output_dim = y_train.shape[1]        # 49

    # Models: (model, X_train_model, X_val_model, d_test_model)
    all_models = {
        "MLP": (
            build_mlp_model(input_dim, output_dim),
            X_train_flat, X_val_flat, d_scaled_flat,
        ),
        "CNN2D": (
            build_cnn2d_model(output_dim),
            X_train_2d, X_val_2d, d_scaled_2d,
        ),
        "DeepONet_MLP_Fourier": (
            build_deeponet_mlp_fourier(input_dim, output_dim),
            X_train_flat, X_val_flat, d_scaled_flat,
        ),
    }

    all_predictions[prior_name] = {}
    all_metrics[prior_name]     = {}
    all_histories[prior_name]   = {}

    for model_name, (model, X_tr_m, X_v_m, d_test) in all_models.items():
        print(f"\n  Training PINN-{model_name} ...")

        history = train_physics_informed(
            model,
            X_train_flat, X_tr_m,
            X_val_flat,   X_v_m,
            y_train, y_val,
            d_orig_train, d_orig_val,
            scaler_m_mean_np, scaler_m_scale_np,
        )

        m_pred_scaled   = model.predict(d_test, verbose=0)
        m_pred_unscaled = scaler_m.inverse_transform(
            m_pred_scaled.reshape(1, -1)).ravel()

        all_predictions[prior_name][model_name] = m_pred_unscaled
        all_metrics[prior_name][model_name] = {
            "RMSE":   rmse_fn(m_pred_unscaled, m_true),
            "Rel_L2": rel_l2(m_pred_unscaled,  m_true),
        }
        all_histories[prior_name][model_name] = history

        print(
            f"    RMSE   = {all_metrics[prior_name][model_name]['RMSE']:.6f}  "
            f"Rel_L2 = {all_metrics[prior_name][model_name]['Rel_L2']:.4f}"
        )

    all_predictions[prior_name]["Gauss-Newton"] = m_GN
    all_predictions[prior_name]["Sample_Mean"]  = m_mean_prior
    all_metrics[prior_name]["Gauss-Newton"] = {
        "RMSE":   rmse_fn(m_GN,        m_true),
        "Rel_L2": rel_l2(m_GN,         m_true),
    }
    all_metrics[prior_name]["Sample_Mean"] = {
        "RMSE":   rmse_fn(m_mean_prior, m_true),
        "Rel_L2": rel_l2(m_mean_prior,  m_true),
    }


# ══════════════════════════════════════════════════════════════════════
# Save CSVs
# ══════════════════════════════════════════════════════════════════════
rows_predictions = []
for prior_name, model_dict in all_predictions.items():
    for model_name, m_pred in model_dict.items():
        for i, val in enumerate(m_pred):
            rows_predictions.append({
                "cell_index": i,
                "prior":      prior_name,
                "model":      model_name,
                "prediction": float(val),
            })
    for i, val in enumerate(m_true):
        rows_predictions.append({
            "cell_index": i,
            "prior":      prior_name,
            "model":      "True",
            "prediction": float(val),
        })

pd.DataFrame(rows_predictions).to_csv(
    os.path.join(RESULTS_DIR, "seismic_2D_pinn_predictions.csv"), index=False
)

rows_metrics = []
for prior_name, model_dict in all_metrics.items():
    for model_name, metrics in model_dict.items():
        rows_metrics.append({
            "prior":   prior_name,
            "model":   model_name,
            "RMSE":    metrics["RMSE"],
            "Rel_L2":  metrics["Rel_L2"],
        })

pd.DataFrame(rows_metrics).to_csv(
    os.path.join(RESULTS_DIR, "seismic_2D_pinn_metrics.csv"), index=False
)

rows_hist = []
for prior_name, model_dict in all_histories.items():
    for model_name, history in model_dict.items():
        for epoch in range(len(history["loss"])):
            rows_hist.append({
                "prior":         prior_name,
                "model":         model_name,
                "epoch":         epoch,
                "loss":          history["loss"][epoch],
                "val_loss":      history["val_loss"][epoch],
                "mse_m":         history["mse_m"][epoch],
                "phys_loss":     history["phys_loss"][epoch],
                "val_mse_m":     history["val_mse_m"][epoch],
                "val_phys_loss": history["val_phys_loss"][epoch],
            })

pd.DataFrame(rows_hist).to_csv(
    os.path.join(RESULTS_DIR, "seismic_2D_pinn_histories.csv"), index=False
)

print("\nSaved files:")
print(os.path.join(RESULTS_DIR, "seismic_2D_pinn_predictions.csv"))
print(os.path.join(RESULTS_DIR, "seismic_2D_pinn_metrics.csv"))
print(os.path.join(RESULTS_DIR, "seismic_2D_pinn_histories.csv"))

print(f"\n{'Prior':<12} {'Model':<25} {'RMSE':>12} {'Rel_L2':>8}")
print("-" * 61)
for prior_name, model_dict in all_metrics.items():
    for model_name, m in model_dict.items():
        print(
            f"{prior_name:<12} {model_name:<25} "
            f"{m['RMSE']:>12.6f} {m['Rel_L2']:>8.4f}"
        )