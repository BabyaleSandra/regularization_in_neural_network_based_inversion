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
RESULTS_DIR = "RESULTS_NN"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Problem geometry ──────────────────────────────────────────────────
SCALE       = 1600
PSCALE      = 7
vbackground = 2900.0
n_param     = PSCALE * PSCALE    # 49
n_obs       = PSCALE * PSCALE    # 49

xn = np.linspace(-150, SCALE + 150, PSCALE)
zn = xn.copy()

# Prior mean in physical units
m0 = (1.0 / vbackground) * np.ones(n_param)   # ≈ 3.448e-04 s/m

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
epochs      = 1000
batch_size  = 256
lr          = 1e-3
l2_strength = 1e-4
dropout     = 0.2
Patience    = 40
wd          = 1e-4
clipnorm    = 1.0

# MLP
hidden_units = (128, 256, 512, 256, 128)

# CNN2D — encoder filters per block
cnn2d_filters = [32, 64, 128]   # three conv blocks
cnn2d_kernel  = (3, 3)
cnn2d_dense   = [256, 128]      # dense head after flatten

# DeepONet
p       = 128
width   = 256
depth   = 4
n_freqs = 16


# ══════════════════════════════════════════════════════════════════════
# Callbacks
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
# MLP — linear output
# Flattened (49,) input → [128→256→512→256→128] + residual → (49,)
# ══════════════════════════════════════════════════════════════════════
def build_mlp_model(input_dim, output_dim):
    inputs = keras.Input(shape=(input_dim,), name="d_flat")
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

    residual = layers.Dense(
        hidden_units[-1], use_bias=False, activation=None,
    )(e1)
    x = layers.Add(name="res_add")([x, residual])
    outputs = layers.Dense(output_dim, activation="linear", name="m")(x)
    return keras.Model(inputs, outputs, name="MLP")


# ══════════════════════════════════════════════════════════════════════
# CNN2D — encoder-decoder, linear output
#
# Input  : (7, 7, 1)  travel-time matrix as single-channel 2D image
#           ttobs[i,j] = travel time source-i → receiver-j
#
# Encoder: three Conv2D blocks (padding=same so spatial dims stay 7×7)
#          each block: Conv2D → BatchNorm → GELU → Dropout
#
# Head   : Flatten → Dense(256) → Dense(128) → Dense(49, linear)
#
# padding=same keeps spatial dims at (7,7) throughout the encoder
# because the grid is already small (7×7 = 49 nodes) — pooling would
# collapse spatial information too aggressively.
# ══════════════════════════════════════════════════════════════════════
def build_cnn2d_model(output_dim):
    # Input: (7, 7, 1) — travel time matrix reshaped as 2D image
    inp = keras.Input(shape=(PSCALE, PSCALE, 1), name="d_2d")
    x   = inp

    # Encoder blocks
    for n_filt in cnn2d_filters:
        x = layers.Conv2D(
            n_filt, cnn2d_kernel, padding="same", activation=None,
            kernel_regularizer=keras.regularizers.l2(l2_strength),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("gelu")(x)
        x = layers.Conv2D(
            n_filt, cnn2d_kernel, padding="same", activation=None,
            kernel_regularizer=keras.regularizers.l2(l2_strength),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("gelu")(x)
        x = layers.Dropout(dropout)(x)
    # Shape still (7, 7, 128) — no pooling to preserve 7×7 resolution

    # Dense head
    x = layers.Flatten()(x)                        # (7*7*128,) = (6272,)
    for units in cnn2d_dense:
        x = layers.Dense(
            units, activation=None,
            kernel_regularizer=keras.regularizers.l2(l2_strength),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("gelu")(x)
        x = layers.Dropout(dropout)(x)

    out = layers.Dense(output_dim, activation="linear", name="m")(x)
    return keras.Model(inp, out, name="CNN2D")


# ══════════════════════════════════════════════════════════════════════
# DeepONet — MLP branch on flattened (49,) observations
#           + Fourier trunk on 2D (x,z) query points
# Linear output — no softplus
# Trunk fix: proper keras.Model with Input so weights are trained
# ══════════════════════════════════════════════════════════════════════
def build_deeponet_mlp_fourier(branch_dim, n_z):
    # 2D Fourier features of node positions in Fortran order
    x_norm = (xn - xn.min()) / (xn.max() - xn.min())
    z_norm = (zn - zn.min()) / (zn.max() - zn.min())
    idx    = np.arange(n_z)
    px     = x_norm[idx % PSCALE]                        # (49,)
    pz     = z_norm[idx // PSCALE]                       # (49,)

    cols = [px.reshape(-1, 1), pz.reshape(-1, 1)]
    for k in range(1, n_freqs + 1):
        cols.append(np.sin(2.0 * np.pi * k * px).reshape(-1, 1))
        cols.append(np.cos(2.0 * np.pi * k * px).reshape(-1, 1))
        cols.append(np.sin(2.0 * np.pi * k * pz).reshape(-1, 1))
        cols.append(np.cos(2.0 * np.pi * k * pz).reshape(-1, 1))
    phi       = np.concatenate(cols, axis=1).astype(np.float32)  # (49, 2+4*n_freqs)
    phi_const = tf.constant(phi, dtype=tf.float32)

    phi_dim   = 2 + 4 * n_freqs
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
    )(t_x)
    trunk_model = keras.Model(trunk_inp, t_x, name="trunk_fourier_2d")

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
    )(b)

    T = layers.Lambda(
        lambda b_dummy: trunk_model(phi_const, training=False),
        name="trunk_eval",
    )(b)

    out = layers.Lambda(
        lambda inputs: tf.einsum("bp,jp->bj", inputs[0], inputs[1]),
        name="m_linear",
    )([b, T])

    return keras.Model(branch_inp, out, name="DeepONet_MLP_Fourier")


# ══════════════════════════════════════════════════════════════════════
# Compile helper
# ══════════════════════════════════════════════════════════════════════
def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.experimental.AdamW(
            learning_rate=lr, weight_decay=wd, clipnorm=clipnorm,
        ),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ══════════════════════════════════════════════════════════════════════
# Error metrics
# ══════════════════════════════════════════════════════════════════════
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
    print(f"  Prior: {prior_name}")
    print(f"{'='*60}")

    M  = np.loadtxt(os.path.join(DATA_DIR, m_file), delimiter=",")
    Gz = np.loadtxt(os.path.join(DATA_DIR, d_file), delimiter=",")

    if M.ndim  == 3: M  = M.reshape(M.shape[0],  -1, order='F')
    if Gz.ndim == 3: Gz = Gz.reshape(Gz.shape[0], -1, order='F')

    m_mean_prior = np.mean(M, axis=0)

    # ── Scaling ───────────────────────────────────────────────────────
    scaler_G  = StandardScaler()
    Gz_scaled = scaler_G.fit_transform(Gz)           # (N, 49) flat

    scaler_m  = StandardScaler()
    M_scaled  = scaler_m.fit_transform(M)            # (N, 49)

    # Single test observation — flat (1, 49) for MLP/DeepONet
    # and reshaped (1, 7, 7, 1) for CNN2D
    d_scaled_flat = scaler_G.transform(d_obs.reshape(1, -1))  # (1, 49)
    d_scaled_2d   = d_scaled_flat.reshape(1, PSCALE, PSCALE, 1)

    # ── Train / val split ─────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        Gz_scaled, M_scaled, test_size=0.15, random_state=42,
    )

    input_dim  = X_train.shape[1]   # 49
    output_dim = y_train.shape[1]   # 49

    # CNN2D needs (N, 7, 7, 1) shaped inputs
    X_train_2d = X_train.reshape(-1, PSCALE, PSCALE, 1)
    X_val_2d   = X_val.reshape(-1,   PSCALE, PSCALE, 1)

    # ── Build models ──────────────────────────────────────────────────
    all_models = {
        "MLP":    (compile_model(build_mlp_model(input_dim, output_dim)),
                   X_train, X_val, d_scaled_flat),
        "CNN2D":  (compile_model(build_cnn2d_model(output_dim)),
                   X_train_2d, X_val_2d, d_scaled_2d),
        "DeepONet_MLP_Fourier": (
            compile_model(build_deeponet_mlp_fourier(input_dim, output_dim)),
            X_train, X_val, d_scaled_flat),
    }

    all_predictions[prior_name] = {}
    all_metrics[prior_name]     = {}
    all_histories[prior_name]   = {}

    for model_name, (model, X_tr, X_v, d_test) in all_models.items():
        print(f"\n  Training {model_name} ...")

        history = model.fit(
            X_tr, y_train,
            validation_data=(X_v, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=make_callbacks(),
            shuffle=True,
            verbose=0,
        )

        # ── Predict ───────────────────────────────────────────────────
        m_pred_scaled   = model.predict(d_test, verbose=0)          # (1, 49)
        m_pred_unscaled = scaler_m.inverse_transform(m_pred_scaled).ravel()

        all_predictions[prior_name][model_name] = m_pred_unscaled
        all_metrics[prior_name][model_name] = {
            "RMSE":   rmse_fn(m_pred_unscaled, m_true),
            "Rel_L2": rel_l2(m_pred_unscaled,  m_true),
        }
        all_histories[prior_name][model_name] = {
            "loss":     history.history["loss"],
            "val_loss": history.history["val_loss"],
        }

        print(
            f"    RMSE   = {all_metrics[prior_name][model_name]['RMSE']:.6f}  "
            f"Rel_L2 = {all_metrics[prior_name][model_name]['Rel_L2']:.4f}"
        )

    # ── Baselines ─────────────────────────────────────────────────────
    all_predictions[prior_name]["Gauss-Newton"] = m_GN
    all_predictions[prior_name]["Sample_Mean"]  = m_mean_prior

    all_metrics[prior_name]["Gauss-Newton"] = {
        "RMSE":   rmse_fn(m_GN,         m_true),
        "Rel_L2": rel_l2(m_GN,          m_true),
    }
    all_metrics[prior_name]["Sample_Mean"] = {
        "RMSE":   rmse_fn(m_mean_prior,  m_true),
        "Rel_L2": rel_l2(m_mean_prior,   m_true),
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
    os.path.join(RESULTS_DIR, "seismic_2D_predictions.csv"), index=False
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
    os.path.join(RESULTS_DIR, "seismic_2D_metrics.csv"), index=False
)

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
    os.path.join(RESULTS_DIR, "seismic_2D_histories.csv"), index=False
)

print("\nSaved files:")
print(os.path.join(RESULTS_DIR, "seismic_2D_predictions.csv"))
print(os.path.join(RESULTS_DIR, "seismic_2D_metrics.csv"))
print(os.path.join(RESULTS_DIR, "seismic_2D_histories.csv"))

print(f"\n{'Prior':<12} {'Model':<25} {'RMSE':>12} {'Rel_L2':>8}")
print("-" * 61)
for prior_name, model_dict in all_metrics.items():
    for model_name, m in model_dict.items():
        print(
            f"{prior_name:<12} {model_name:<25} "
            f"{m['RMSE']:>12.6f} {m['Rel_L2']:>8.4f}"
        )