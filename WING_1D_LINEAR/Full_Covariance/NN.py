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
RESULTS_DIR = "RESULTS_NN"          # NN with softplus
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── parameters ───────────
n          = 50                                  # parameter grid size
m          = 20                                  # observation grid size
t          = np.linspace(0, 1, n)               # parameter grid
s          = np.linspace(0, 1, m)               # observation grid
dt         = t[1] - t[0]
sigma_d    = 0.01                               # In[2]
sigma_z    = 1                                  # In[7]
Delta      = 0.02                               

# ── Forward operator G (In[3]: build_G) ──────────────────────────────
# G[i,j] = t[j] * exp(-s[i]*t[j]^2) * dt   shape (m, n) = (20, 50)
def build_G(s_grid, t_grid):
    S, T = np.meshgrid(s_grid, t_grid, indexing="ij")
    return T * np.exp(-S * T**2) * (t_grid[1] - t_grid[0])

G_mat = build_G(s, t)                           # (20, 50) — saved for reference

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

# ── Hyperparameters (matching NN_no_softplus.ipynb In[6]) ────────────
epochs      = 1000
batch_size  = 256
lr          = 1e-3
l2_strength = 1e-4
dropout     = 0.2
Patience    = 40
wd          = 1e-4
clipnorm    = 1.0

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
# Callbacks  (In[6])
# ══════════════════════════════════════════════════════════════════════
def make_callbacks():
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=Patience,
            restore_best_weights=True,
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=15,
            min_lr=1e-6,
            verbose=0,
        ),
    ]


# ══════════════════════════════════════════════════════════════════════
# MLP  (In[7] — softplus output)
# d(input_dim) -> [128->256->512->256->128] + residual -> softplus
# Each block: Dense(linear) -> BatchNorm -> GELU -> Dropout
# Residual: linear projection of first hidden output added to last.
# Output: Dense(linear) -> Activation("softplus")  [decoupled]
# ══════════════════════════════════════════════════════════════════════
def build_mlp_model(input_dim, output_dim):
    inputs = keras.Input(shape=(input_dim,), name="d")
    x  = inputs
    e1 = None

    for i, units in enumerate(hidden_units):
        x = layers.Dense(
            units, activation=None,
            kernel_regularizer=keras.regularizers.l2(l2_strength),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("gelu")(x)
        x = layers.Dropout(dropout)(x)
        if i == 0:
            e1 = x

    # Residual: project first hidden (128) -> last hidden width (128)
    residual = layers.Dense(
        hidden_units[-1], use_bias=False, activation=None,
    )(e1)
    x = layers.Add(name="res_add")([x, residual])

    # Output: linear Dense then softplus as separate layer
    x       = layers.Dense(output_dim, activation=None, name="z_linear")(x)
    outputs = layers.Activation("softplus", name="z_softplus")(x)
    return keras.Model(inputs, outputs, name="MLP")


# ══════════════════════════════════════════════════════════════════════
# CNN  (In[8] — softplus output)
# Parallel Conv1D branches (kernels 1,3,5,7), GlobalAveragePooling,
# dense head (512,256), linear Dense -> softplus.
# ══════════════════════════════════════════════════════════════════════
def build_cnn_model(input_dim, output_dim):
    inp = keras.Input(shape=(input_dim,))
    x   = layers.Reshape((input_dim, 1), name="d")(inp)

    def one_branch(k):
        b = layers.Conv1D(
            n_filters, k, padding="same", activation=None,
            kernel_regularizer=keras.regularizers.l2(l2_strength),
        )(x)
        b = layers.BatchNormalization()(b)
        b = layers.Activation("gelu")(b)
        b = layers.Conv1D(
            n_filters, k, padding="same", activation=None,
            kernel_regularizer=keras.regularizers.l2(l2_strength),
        )(b)
        b = layers.BatchNormalization()(b)
        b = layers.Activation("gelu")(b)
        return layers.GlobalAveragePooling1D()(b)

    feat = [one_branch(k) for k in kernels]
    h    = layers.Concatenate()(feat)              # (B, 4*n_filters)

    for units in dense_units:
        h = layers.Dense(
            units, activation=None,
            kernel_regularizer=keras.regularizers.l2(l2_strength),
        )(h)
        h = layers.BatchNormalization()(h)
        h = layers.Activation("gelu")(h)
        h = layers.Dropout(dropout)(h)

    # Output: linear Dense then softplus as separate layer
    h   = layers.Dense(output_dim, activation=None, name="z_linear")(h)
    out = layers.Activation("softplus", name="z_softplus")(h)
    return keras.Model(inp, out, name="CNN")


# ══════════════════════════════════════════════════════════════════════
# DeepONet — MLP branch + Fourier trunk  (softplus output)
#
# FIX: trunk is a proper keras.Model with its own Input so trunk
# weights receive gradients (frozen-trunk bug from notebook fixed).
#
# Query coordinate is t in [0,1] — Fourier features of the fixed
# t-grid, same construction as gravity but over [0,1] not [0,100].
# Output: einsum field -> softplus
# ══════════════════════════════════════════════════════════════════════
def build_deeponet_mlp_fourier(branch_dim, n_z):
    """
    branch_dim : m = 20  (observation dimension)
    n_z        : n = 50  (parameter dimension)
    """
    # Fourier features of fixed t-grid (already in [0,1])
    t_norm = np.linspace(0.0, 1.0, n_z, dtype=np.float32).reshape(-1, 1)
    cols   = [t_norm]
    for k in range(1, n_freqs + 1):
        cols.append(np.sin(2.0 * np.pi * k * t_norm))
        cols.append(np.cos(2.0 * np.pi * k * t_norm))
    phi       = np.concatenate(cols, axis=1)        # (n_z, 1+2*n_freqs)
    phi_const = tf.constant(phi, dtype=tf.float32)

    # ── Trunk sub-model ───────────────────────────────────────────────
    phi_dim   = 1 + 2 * n_freqs
    trunk_inp = keras.Input(shape=(phi_dim,), name="trunk_input")
    t_x = trunk_inp
    for _ in range(depth):
        t_x = layers.Dense(
            width, activation=None,
            kernel_regularizer=keras.regularizers.l2(l2_strength),
        )(t_x)
        t_x = layers.BatchNormalization()(t_x)
        t_x = layers.Activation("gelu")(t_x)
        t_x = layers.Dropout(dropout)(t_x)
    t_x = layers.Dense(
        p, activation="linear",
        kernel_regularizer=keras.regularizers.l2(l2_strength),
        name="trunk_out",
    )(t_x)                                          # (n_z, p)
    trunk_model = keras.Model(trunk_inp, t_x, name="trunk_fourier")

    # ── Branch sub-model ──────────────────────────────────────────────
    branch_inp = keras.Input(shape=(branch_dim,), name="branch_input")
    b = branch_inp
    for _ in range(depth):
        b = layers.Dense(
            width, activation=None,
            kernel_regularizer=keras.regularizers.l2(l2_strength),
        )(b)
        b = layers.BatchNormalization()(b)
        b = layers.Activation("gelu")(b)
        b = layers.Dropout(dropout)(b)
    b = layers.Dense(
        p, activation="linear",
        kernel_regularizer=keras.regularizers.l2(l2_strength),
        name="branch_out",
    )(b)                                            # (B, p)

    # Trunk evaluated inside graph (trunk fix)
    T = layers.Lambda(
        lambda b_dummy: trunk_model(phi_const, training=False),
        name="trunk_eval",
    )(b)                                            # (n_z, p)

    # Combine: (B,p) x (n_z,p)^T -> (B, n_z) then softplus
    field = layers.Lambda(
        lambda inputs: tf.einsum("bp,jp->bj", inputs[0], inputs[1]),
        name="deeponet_field",
    )([b, T])                                       # (B, n_z)

    out = layers.Activation("softplus", name="z_softplus")(field)
    return keras.Model(branch_inp, out, name="DeepONet_MLP_Fourier")


# ══════════════════════════════════════════════════════════════════════
# Compile helper
# ══════════════════════════════════════════════════════════════════════
def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.experimental.AdamW(
            learning_rate=lr,
            weight_decay=wd,
            clipnorm=clipnorm,
        ),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ══════════════════════════════════════════════════════════════════════
# Error metrics  (In[22] of NN_no_softplus.ipynb)
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
    print(f"  Prior: {prior_name}")
    print(f"{'='*60}")

    # ── Load prior samples ────────────────────────────────────────────
    # Z  : (n_samples, n=50)  — parameter samples from this prior
    # Gz : (n_samples, m=20)  — corresponding noisy observations
    Z  = np.loadtxt(os.path.join(DATA_DIR, z_file), delimiter=",")
    Gz = np.loadtxt(os.path.join(DATA_DIR, d_file), delimiter=",")

    z_mean_prior = np.mean(Z, axis=0)

    # ── Scaling ───────────────────────────────────────────────────────
    # scaler_G : normalise observations (NN input)
    # scaler_z : normalise parameters   (NN output)
    # FIX: store scaled targets in Z_scaled (not z_scaled) to avoid
    # variable name collision in prediction loop.
    scaler_G  = StandardScaler()
    Gz_scaled = scaler_G.fit_transform(Gz)           # (N, 20)

    scaler_z  = StandardScaler()
    Z_scaled  = scaler_z.fit_transform(Z)            # (N, 50)

    # Transform single test observation with fitted G-scaler
    d_scaled = scaler_G.transform(d_obs.reshape(1, -1))  # (1, 20)

    # ── Train / val split ─────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        Gz_scaled, Z_scaled,
        test_size=0.15,
        random_state=42,
    )

    input_dim  = X_train.shape[1]   # 20
    output_dim = y_train.shape[1]   # 50

    # ── Build models — fresh weights each prior ───────────────────────
    all_models = {
        "MLP": compile_model(build_mlp_model(input_dim, output_dim)),
        "CNN": compile_model(build_cnn_model(input_dim, output_dim)),
        "DeepONet_MLP_Fourier": compile_model(
            build_deeponet_mlp_fourier(input_dim, output_dim)
        ),
    }

    all_predictions[prior_name] = {}
    all_metrics[prior_name]     = {}
    all_histories[prior_name]   = {}

    for model_name, model in all_models.items():
        print(f"\n  Training {model_name} ...")

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=make_callbacks(),
            shuffle=True,
            verbose=0,
        )

        # ── Predict ───────────────────────────────────────────────────
        # FIX: z_pred_scaled / z_pred_unscaled never reuse Z_scaled.
        z_pred_scaled   = model.predict(d_scaled, verbose=0)          # (1,50)
        z_pred_unscaled = scaler_z.inverse_transform(z_pred_scaled).ravel()

        all_predictions[prior_name][model_name] = z_pred_unscaled
        all_metrics[prior_name][model_name] = {
            "RMSE":   rmse_fn(z_pred_unscaled, z_true),
            "Rel_L2": rel_l2(z_pred_unscaled,  z_true),
        }
        all_histories[prior_name][model_name] = {
            "loss":     history.history["loss"],
            "val_loss": history.history["val_loss"],
        }

        print(
            f"    RMSE   = {all_metrics[prior_name][model_name]['RMSE']:.4f}  "
            f"Rel_L2 = {all_metrics[prior_name][model_name]['Rel_L2']:.4f}"
        )

    # ── Baselines — MAP and sample mean ──────────────────────────────
    # Wing uses MAP (Gaussian posterior mean) not Gauss-Newton
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
# Save CSVs — wing schema (t_index / t columns, not w_index / w)
# Compatible with ANALYSIS_WING_PAPER.py
# ══════════════════════════════════════════════════════════════════════

# ── wing_predictions.csv ──────────────────────────────────────────────
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
    # True solution stored once per prior block for easy subsetting
    for i, val in enumerate(z_true):
        rows_predictions.append({
            "t_index":    i,
            "t":          t[i],
            "prior":      prior_name,
            "model":      "True",
            "prediction": float(val),
        })

pd.DataFrame(rows_predictions).to_csv(
    os.path.join(RESULTS_DIR, "wing_predictions.csv"), index=False
)

# ── wing_metrics.csv ──────────────────────────────────────────────────
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
    os.path.join(RESULTS_DIR, "wing_metrics.csv"), index=False
)

# ── wing_histories.csv ────────────────────────────────────────────────
# Plain MSE training: loss and val_loss only.
# mse_z / phys_loss columns absent (unlike PINN histories).
rows_hist = []
for prior_name, model_dict in all_histories.items():
    for model_name, history in model_dict.items():
        for epoch in range(len(history["loss"])):
            rows_hist.append({
                "prior":    prior_name,
                "model":    model_name,
                "epoch":    epoch,
                "loss":     history["loss"][epoch],
                "val_loss": history["val_loss"][epoch],
            })

pd.DataFrame(rows_hist).to_csv(
    os.path.join(RESULTS_DIR, "wing_histories.csv"), index=False
)

print("\nSaved files:")
print(os.path.join(RESULTS_DIR, "wing_predictions.csv"))
print(os.path.join(RESULTS_DIR, "wing_metrics.csv"))
print(os.path.join(RESULTS_DIR, "wing_histories.csv"))

# ── Console summary ───────────────────────────────────────────────────
print(f"\n{'Prior':<12} {'Model':<25} {'RMSE':>8} {'Rel_L2':>8}")
print("-" * 57)
for prior_name, model_dict in all_metrics.items():
    for model_name, m in model_dict.items():
        print(
            f"{prior_name:<12} {model_name:<25} "
            f"{m['RMSE']:>8.4f} {m['Rel_L2']:>8.4f}"
        )