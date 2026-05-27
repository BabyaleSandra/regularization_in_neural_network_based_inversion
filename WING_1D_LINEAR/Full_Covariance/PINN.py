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

# ── Reproducibility ───────────────────────────────────────────────────
SEED = 42
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

# ── Directories ───────────────────────────────────────────────────────
DATA_DIR    = "DATA"
RESULTS_DIR = "RESULTS_PINN"          # PINN with softplus
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Problem geometry (generate_data.ipynb) ───────────
n          = 50
m          = 20
t          = np.linspace(0, 1, n)
s          = np.linspace(0, 1, m)
dt         = t[1] - t[0]
sigma_d    = 0.01
sigma_z    = 1
Delta      = 0.02 

# ── Forward operator G (In[3]: build_G) ──────────────────────────────
# G[i,j] = t[j] * exp(-s[i]*t[j]^2) * dt   shape (m, n) = (20, 50)
def build_G(s_grid, t_grid):
    S, T = np.meshgrid(s_grid, t_grid, indexing="ij")
    return T * np.exp(-S * T**2) * (t_grid[1] - t_grid[0])

G_np = build_G(s, t)                            # (20, 50) numpy
G_tf = tf.constant(G_np, dtype=tf.float32)      # (20, 50) tf constant

# ── Load fixed reference data ─────────────────────────────────────────
z_true = np.loadtxt(os.path.join(DATA_DIR, "z_true.csv"),    delimiter=",")
d_obs  = np.loadtxt(os.path.join(DATA_DIR, "d_obs.csv"),     delimiter=",")
z_MAP  = np.loadtxt(os.path.join(DATA_DIR, "z_MAP.csv"), delimiter=",")

# ── Prior file map ────────────────────────────────────────────────────
PRIORS = {
    "Gaussian": ("z_gaussian.csv", "d_gaussian.csv"),
    "Laplace":  ("z_laplace.csv",  "d_laplace.csv"),
    "TV":       ("z_tv.csv",       "d_tv.csv"),
    "Uniform":  ("z_uniform.csv",  "d_uniform.csv"),
}

# ── Hyperparameters (matching PINN_softplus.ipynb In[6]) ─────────────
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

# MLP (In[7])
hidden_units = (128, 256, 512, 256, 128)

# CNN (In[8])
kernels     = (1, 3, 5, 7)
n_filters   = 64
dense_units = (512, 256)

# DeepONet MLP-Fourier (In[9-12])
p       = 128
width   = 256
depth   = 4
n_freqs = 16


# ══════════════════════════════════════════════════════════════════════
# Physics-informed loss  (PINN_softplus.ipynb In[13])
#
# Wing forward operator is linear: u = G @ z
# So the physics residual is simply:
#     L_phys = MSE(G_tf @ z_pred_phys,  d_orig_batch)
#
# d_orig_batch is the UNSCALED observation paired with each training
# sample — NOT the single fixed test observation d_obs.
# z_pred_phys is floored at 1e-4 (safety, matching gravity convention).
# ══════════════════════════════════════════════════════════════════════
def make_physics_loss(z_mean_tf, z_std_tf):
    """Closure over per-prior scaler constants."""
    def physics_informed_loss(y_batch_scaled, z_pred_scaled, d_orig_batch):
        # Supervised MSE in scaled space
        mse_z = tf.reduce_mean((y_batch_scaled - z_pred_scaled) ** 2)

        # Unscale z_pred to physical units
        z_pred_phys = z_pred_scaled * z_std_tf + z_mean_tf  # (B, 50)
        z_pred_phys = tf.maximum(z_pred_phys, 1e-4)         # safety floor

        # Physics residual: linear forward operator G @ z
        # G_tf : (20, 50),  z_pred_phys : (B, 50)
        # matmul(z_pred_phys, G_tf^T) -> (B, 20)
        u_pred = tf.linalg.matmul(z_pred_phys, G_tf, transpose_b=True)
        phys   = tf.reduce_mean((u_pred - d_orig_batch) ** 2)

        total  = mse_z + lambda_phys * phys
        return total, mse_z, phys

    return physics_informed_loss


# ══════════════════════════════════════════════════════════════════════
# Train / val step factories  (PINN_softplus.ipynb In[13])
# ══════════════════════════════════════════════════════════════════════
def make_train_step(model, optimizer, loss_fn):
    @tf.function
    def train_step(X_batch, y_batch, d_orig_batch):
        with tf.GradientTape() as tape:
            z_pred            = model(X_batch, training=True)
            loss, mse_z, phys = loss_fn(y_batch, z_pred, d_orig_batch)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, mse_z, phys
    return train_step


def make_val_step(model, loss_fn):
    @tf.function
    def val_step(X_batch, y_batch, d_orig_batch):
        z_pred            = model(X_batch, training=False)
        loss, mse_z, phys = loss_fn(y_batch, z_pred, d_orig_batch)
        return loss, mse_z, phys
    return val_step


# ══════════════════════════════════════════════════════════════════════
# Custom training loop  (PINN_softplus.ipynb In[14])
#
# Faithful to PINN_softplus.ipynb:
#   - d_orig shuffled in lockstep with X and y
#   - early stopping on TOTAL val_loss (supervised + physics)
#   - ReduceLROnPlateau on TOTAL val_loss, plateau_patience=15
# ══════════════════════════════════════════════════════════════════════
def train_physics_informed(model, X_train, y_train, X_val, y_val,
                           d_orig_train_np, d_orig_val_np,
                           scaler_z_mean_np, scaler_z_scale_np):

    optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=lr, weight_decay=wd, clipnorm=clipnorm
    )

    z_mean_tf = tf.constant(scaler_z_mean_np,  dtype=tf.float32)
    z_std_tf  = tf.constant(scaler_z_scale_np, dtype=tf.float32)

    loss_fn    = make_physics_loss(z_mean_tf, z_std_tf)
    train_step = make_train_step(model, optimizer, loss_fn)
    val_step   = make_val_step(model, loss_fn)

    # Convert to tf.constant once
    X_tr = tf.constant(X_train,         dtype=tf.float32)
    y_tr = tf.constant(y_train,         dtype=tf.float32)
    d_tr = tf.constant(d_orig_train_np, dtype=tf.float32)
    X_v  = tf.constant(X_val,           dtype=tf.float32)
    y_v  = tf.constant(y_val,           dtype=tf.float32)
    d_v  = tf.constant(d_orig_val_np,   dtype=tf.float32)

    N     = X_tr.shape[0]
    steps = int(np.ceil(N / batch_size))

    best_val         = np.inf
    best_weights     = model.get_weights()
    patience_counter = 0
    plateau_counter  = 0
    current_lr       = lr

    history = {
        "loss": [], "val_loss": [],
        "mse_z": [], "phys_loss": [],
        "val_mse_z": [], "val_phys_loss": [],
    }

    for epoch in range(epochs):
        # Shuffle X, y, d_orig in lockstep
        idx  = tf.random.shuffle(tf.range(N))
        X_sh = tf.gather(X_tr, idx)
        y_sh = tf.gather(y_tr, idx)
        d_sh = tf.gather(d_tr, idx)

        ep_loss, ep_mse, ep_phys = [], [], []

        for step in range(steps):
            Xb = X_sh[step * batch_size : (step + 1) * batch_size]
            yb = y_sh[step * batch_size : (step + 1) * batch_size]
            db = d_sh[step * batch_size : (step + 1) * batch_size]
            l, mz, ph = train_step(Xb, yb, db)
            ep_loss.append(float(l))
            ep_mse.append(float(mz))
            ep_phys.append(float(ph))

        # Validation on full val set
        val_loss, val_mse, val_phys = val_step(X_v, y_v, d_v)
        val_loss = float(val_loss)
        val_mse  = float(val_mse)
        val_phys = float(val_phys)

        history["loss"].append(float(np.mean(ep_loss)))
        history["val_loss"].append(val_loss)
        history["mse_z"].append(float(np.mean(ep_mse)))
        history["phys_loss"].append(float(np.mean(ep_phys)))
        history["val_mse_z"].append(val_mse)
        history["val_phys_loss"].append(val_phys)

        # Early stopping on TOTAL val_loss
        if val_loss < best_val:
            best_val         = val_loss
            best_weights     = model.get_weights()
            patience_counter = 0
            plateau_counter  = 0
        else:
            patience_counter += 1
            plateau_counter  += 1

        # ReduceLROnPlateau on TOTAL val_loss
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
# Model builders  (identical to NN_wing_all_priors.py)
# Softplus: Dense(linear) -> Activation("softplus")
# ══════════════════════════════════════════════════════════════════════

def build_mlp_model(input_dim, output_dim):
    inputs = keras.Input(shape=(input_dim,), name="d")
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
    x = layers.Dense(output_dim, activation=None, name="z_linear")(x)
    outputs = layers.Activation("softplus", name="z_softplus")(x)
    return keras.Model(inputs, outputs, name="MLP")


def build_cnn_model(input_dim, output_dim):
    inp = keras.Input(shape=(input_dim,))
    x   = layers.Reshape((input_dim, 1), name="d")(inp)

    def one_branch(k):
        b = layers.Conv1D(n_filters, k, padding="same", activation=None,
                          kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
        b = layers.BatchNormalization()(b)
        b = layers.Activation("gelu")(b)
        b = layers.Conv1D(n_filters, k, padding="same", activation=None,
                          kernel_regularizer=keras.regularizers.l2(l2_strength))(b)
        b = layers.BatchNormalization()(b)
        b = layers.Activation("gelu")(b)
        return layers.GlobalAveragePooling1D()(b)

    h = layers.Concatenate()([one_branch(k) for k in kernels])
    for units in dense_units:
        h = layers.Dense(units, activation=None,
                          kernel_regularizer=keras.regularizers.l2(l2_strength))(h)
        h = layers.BatchNormalization()(h)
        h = layers.Activation("gelu")(h)
        h = layers.Dropout(dropout)(h)
    h   = layers.Dense(output_dim, activation=None, name="z_linear")(h)
    out = layers.Activation("softplus", name="z_softplus")(h)
    return keras.Model(inp, out, name="CNN")


def build_deeponet_mlp_fourier(branch_dim, n_z):
    t_norm = np.linspace(0.0, 1.0, n_z, dtype=np.float32).reshape(-1, 1)
    cols   = [t_norm]
    for k in range(1, n_freqs + 1):
        cols.append(np.sin(2.0 * np.pi * k * t_norm))
        cols.append(np.cos(2.0 * np.pi * k * t_norm))
    phi       = np.concatenate(cols, axis=1)
    phi_const = tf.constant(phi, dtype=tf.float32)

    phi_dim   = 1 + 2 * n_freqs
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
    trunk_model = keras.Model(trunk_inp, t_x, name="trunk_fourier")

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

    field = layers.Lambda(
        lambda inputs: tf.einsum("bp,jp->bj", inputs[0], inputs[1]),
        name="deeponet_field")([b, T])

    out = layers.Activation("softplus", name="z_softplus")(field)
    return keras.Model(branch_inp, out, name="DeepONet_MLP_Fourier")


# ══════════════════════════════════════════════════════════════════════
# Error metrics
# ══════════════════════════════════════════════════════════════════════
def rel_l2(z_pred, z_ref):
    return float(np.linalg.norm(z_pred - z_ref) / np.linalg.norm(z_ref))

def rmse_fn(z_pred, z_ref):
    return float(np.sqrt(np.mean((z_pred - z_ref) ** 2)))


# ══════════════════════════════════════════════════════════════════════
# Main training loop — all four priors
# ══════════════════════════════════════════════════════════════════════
all_predictions = {}
all_metrics     = {}
all_histories   = {}

for prior_name, (z_file, d_file) in PRIORS.items():
    print(f"\n{'='*60}")
    print(f"  Prior: {prior_name}  [PINN]")
    print(f"{'='*60}")

    Z  = np.loadtxt(os.path.join(DATA_DIR, z_file), delimiter=",")
    Gz = np.loadtxt(os.path.join(DATA_DIR, d_file), delimiter=",")

    z_mean_prior = np.mean(Z, axis=0)

    # ── Scaling ───────────────────────────────────────────────────────
    scaler_G  = StandardScaler()
    Gz_scaled = scaler_G.fit_transform(Gz)

    scaler_z  = StandardScaler()
    Z_scaled  = scaler_z.fit_transform(Z)

    scaler_z_mean_np  = scaler_z.mean_.astype(np.float32)   # (50,)
    scaler_z_scale_np = scaler_z.scale_.astype(np.float32)  # (50,)

    d_scaled = scaler_G.transform(d_obs.reshape(1, -1))

    # ── Train / val split ─────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        Gz_scaled, Z_scaled, test_size=0.15, random_state=42
    )

    # ── d_orig: unscale X back to physical observations ───────────────
    # d_orig_train[i] is the noisy observation that generated sample i.
    # Shuffled in lockstep with X_train and y_train each epoch.
    d_orig_train = (X_train * scaler_G.scale_ + scaler_G.mean_).astype(np.float32)
    d_orig_val   = (X_val   * scaler_G.scale_ + scaler_G.mean_).astype(np.float32)

    input_dim  = X_train.shape[1]   # 20
    output_dim = y_train.shape[1]   # 50

    # ── Build models — fresh weights each prior ───────────────────────
    all_models = {
        "MLP":                  build_mlp_model(input_dim, output_dim),
        "CNN":                  build_cnn_model(input_dim, output_dim),
        "DeepONet_MLP_Fourier": build_deeponet_mlp_fourier(input_dim,
                                                            output_dim),
    }

    all_predictions[prior_name] = {}
    all_metrics[prior_name]     = {}
    all_histories[prior_name]   = {}

    for model_name, model in all_models.items():
        print(f"\n  Training PINN-{model_name} ...")

        history = train_physics_informed(
            model,
            X_train, y_train, X_val, y_val,
            d_orig_train, d_orig_val,
            scaler_z_mean_np, scaler_z_scale_np,
        )

        # ── Predict ───────────────────────────────────────────────────
        z_pred_scaled   = model.predict(d_scaled, verbose=0)          # (1,50)
        z_pred_unscaled = scaler_z.inverse_transform(z_pred_scaled).ravel()

        all_predictions[prior_name][model_name] = z_pred_unscaled
        all_metrics[prior_name][model_name] = {
            "RMSE":   rmse_fn(z_pred_unscaled, z_true),
            "Rel_L2": rel_l2(z_pred_unscaled,  z_true),
        }
        all_histories[prior_name][model_name] = history

        print(
            f"    RMSE   = {all_metrics[prior_name][model_name]['RMSE']:.4f}  "
            f"Rel_L2 = {all_metrics[prior_name][model_name]['Rel_L2']:.4f}"
        )

    # ── Baselines ─────────────────────────────────────────────────────
    all_predictions[prior_name]["MAP"]         = z_MAP
    all_predictions[prior_name]["Sample_Mean"] = z_mean_prior
    all_metrics[prior_name]["MAP"] = {
        "RMSE":   rmse_fn(z_MAP,         z_true),
        "Rel_L2": rel_l2(z_MAP,          z_true),
    }
    all_metrics[prior_name]["Sample_Mean"] = {
        "RMSE":   rmse_fn(z_mean_prior,  z_true),
        "Rel_L2": rel_l2(z_mean_prior,   z_true),
    }


# ══════════════════════════════════════════════════════════════════════
# Save CSVs — wing schema (t_index / t columns)
# Compatible with ANALYSIS_WING_PAPER.py
# ══════════════════════════════════════════════════════════════════════

# ── wing_pinn_predictions.csv ─────────────────────────────────────────
rows_predictions = []
for prior_name, model_dict in all_predictions.items():
    for model_name, z_pred in model_dict.items():
        for i, val in enumerate(z_pred):
            rows_predictions.append({
                "t_index":    i,
                "t":          t[i],
                "prior":      prior_name,
                "model":      model_name,
                "prediction": float(val),
            })
    for i, val in enumerate(z_true):
        rows_predictions.append({
            "t_index":    i,
            "t":          t[i],
            "prior":      prior_name,
            "model":      "True",
            "prediction": float(val),
        })

pd.DataFrame(rows_predictions).to_csv(
    os.path.join(RESULTS_DIR, "wing_pinn_predictions.csv"), index=False
)

# ── wing_pinn_metrics.csv ─────────────────────────────────────────────
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
    os.path.join(RESULTS_DIR, "wing_pinn_metrics.csv"), index=False
)

# ── wing_pinn_histories.csv ───────────────────────────────────────────
# Includes mse_z, phys_loss, val_mse_z, val_phys_loss unlike plain NN.
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
                "mse_z":         history["mse_z"][epoch],
                "phys_loss":     history["phys_loss"][epoch],
                "val_mse_z":     history["val_mse_z"][epoch],
                "val_phys_loss": history["val_phys_loss"][epoch],
            })

pd.DataFrame(rows_hist).to_csv(
    os.path.join(RESULTS_DIR, "wing_pinn_histories.csv"), index=False
)

print("\nSaved files:")
print(os.path.join(RESULTS_DIR, "wing_pinn_predictions.csv"))
print(os.path.join(RESULTS_DIR, "wing_pinn_metrics.csv"))
print(os.path.join(RESULTS_DIR, "wing_pinn_histories.csv"))

# ── Console summary ───────────────────────────────────────────────────
print(f"\n{'Prior':<12} {'Model':<25} {'RMSE':>8} {'Rel_L2':>8}")
print("-" * 57)
for prior_name, model_dict in all_metrics.items():
    for model_name, m in model_dict.items():
        print(
            f"{prior_name:<12} {model_name:<25} "
            f"{m['RMSE']:>8.4f} {m['Rel_L2']:>8.4f}"
        )