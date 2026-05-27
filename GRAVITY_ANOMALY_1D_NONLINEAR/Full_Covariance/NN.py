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

# ── Problem geometry (Generate_data_1D-FULL.ipynb In[2,3,10]) ─────────
H          = 10.0
n_param    = 100                                 # z-grid size  (In[2]: n=100)
wmin, wmax = 0.0, 100.0
w          = np.linspace(wmin, wmax, n_param)    # parameter grid
n_obs      = 15                                  # In[2]: m=15
x_obs      = np.linspace(wmin, wmax, n_obs)
sigma_d    = 0.1                                 # In[2]
sigma_z    = 1                                 # In[10]
Delta      = 5                                   # In[10]
maxz       = 2.5                                 # In[2]

# ── Load fixed reference data ─────────────────────────────────────────
z_true = np.loadtxt(os.path.join(DATA_DIR, "z_true.csv"),    delimiter=",")
d_obs  = np.loadtxt(os.path.join(DATA_DIR, "u_obs.csv"),     delimiter=",")
z_GN   = np.loadtxt(os.path.join(DATA_DIR, "z_GN.csv"), delimiter=",")

# ── Prior file map ────────────────────────────────────────────────────
PRIORS = {
    "Gaussian": ("z_gaussian.csv", "d_gaussian.csv"),
    "Laplace":  ("z_laplace.csv",  "d_laplace.csv"),
    "TV":       ("z_tv.csv",       "d_tv.csv"),
    "Uniform":  ("z_uniform.csv",  "d_uniform.csv"),
}

# ── Hyperparameters (NN_no_softplus.ipynb In[6]) ──────────────────────
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

# CNN (In[11])
kernels     = (1, 3, 5, 7)
n_filters   = 64
dense_units = (512, 256)

# DeepONet MLP-Fourier (In[15-18], variant "2_mlp_fourier")
p       = 128   # inner dimension
width   = 256   # hidden width  (In[6]: width=256)
depth   = 4     # hidden depth  (In[6]: depth=4)
n_freqs = 16    # Fourier frequencies (In[15]: n_freqs=16)


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
# MLP  (In[7])
# d(input_dim) -> [128->256->512->256->128] + residual -> softplus out
# Each block: Dense(linear) -> BatchNorm -> GELU -> Dropout
# Residual: linear projection of first hidden output added to last.
# Output: Dense(linear) -> Activation("softplus")
# ══════════════════════════════════════════════════════════════════════
def build_mlp_model(input_dim, output_dim):
    inputs = keras.Input(shape=(input_dim,), name="d")
    x  = inputs
    e1 = None

    for i, units in enumerate(hidden_units):
        x = layers.Dense(
            units,
            activation=None,
            kernel_regularizer=keras.regularizers.l2(l2_strength),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("gelu")(x)
        x = layers.Dropout(dropout)(x)
        if i == 0:
            e1 = x

    # Residual: project first hidden (128) -> last hidden width (128)
    residual = layers.Dense(
        hidden_units[-1],
        use_bias=False,
        activation=None,
    )(e1)
    x = layers.Add(name="res_add")([x, residual])

    # Output: linear Dense then softplus as a separate layer
    x = layers.Dense(output_dim, activation=None, name="z_linear")(x)
    outputs = layers.Activation("softplus", name="z_softplus")(x)

    return keras.Model(inputs, outputs, name="MLP")


# ══════════════════════════════════════════════════════════════════════
# CNN  (In[11])
# Parallel Conv1D branches (kernels 1,3,5,7), GlobalAveragePooling,
# dense head (512,256), linear Dense -> softplus output.
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
        b = layers.GlobalAveragePooling1D()(b)
        return b

    feat = [one_branch(k) for k in kernels]
    h    = layers.Concatenate()(feat)              # (B, 4*n_filters = 256)

    for units in dense_units:
        h = layers.Dense(
            units,
            activation=None,
            kernel_regularizer=keras.regularizers.l2(l2_strength),
        )(h)
        h = layers.BatchNormalization()(h)
        h = layers.Activation("gelu")(h)
        h = layers.Dropout(dropout)(h)

    # Output: linear Dense then softplus as a separate layer
    h   = layers.Dense(output_dim, activation=None, name="z_linear")(h)
    out = layers.Activation("softplus", name="z_softplus")(h)

    return keras.Model(inp, out, name="CNN")


# ══════════════════════════════════════════════════════════════════════
# DeepONet — MLP branch + Fourier trunk  (In[15-18], "2_mlp_fourier")
#
# FIX: trunk is a proper Keras sub-model with its own Input so that
# trunk weights receive gradients.  In the notebook trunk_matrix()
# called mlp_stack() on a tf.constant outside the model graph, leaving
# trunk weights frozen at random initialisation.
#
# Output: einsum("bp,jp->bj") [linear] -> softplus
# ══════════════════════════════════════════════════════════════════════
def build_deeponet_mlp_fourier(branch_dim, n_z):
    """
    branch_dim : n_obs   = 15
    n_z        : n_param = 100
    """

    # ── Fourier features of the fixed w-grid (In[15]) ────────────────
    w_norm = np.linspace(0.0, 1.0, n_z, dtype=np.float32).reshape(-1, 1)
    cols   = [w_norm]
    for k in range(1, n_freqs + 1):
        cols.append(np.sin(2.0 * np.pi * k * w_norm))
        cols.append(np.cos(2.0 * np.pi * k * w_norm))
    phi       = np.concatenate(cols, axis=1)         # (n_z, 1+2*n_freqs)
    phi_const = tf.constant(phi, dtype=tf.float32)

    # ── Trunk sub-model ───────────────────────────────────────────────
    phi_dim   = 1 + 2 * n_freqs
    trunk_inp = keras.Input(shape=(phi_dim,), name="trunk_input")
    t = trunk_inp
    for _ in range(depth):
        t = layers.Dense(
            width, activation=None,
            kernel_regularizer=keras.regularizers.l2(l2_strength),
        )(t)
        t = layers.BatchNormalization()(t)
        t = layers.Activation("gelu")(t)
        t = layers.Dropout(dropout)(t)
    t = layers.Dense(
        p, activation=None,
        kernel_regularizer=keras.regularizers.l2(l2_strength),
        name="trunk_out",
    )(t)                                             # (n_z, p)  linear
    trunk_model = keras.Model(trunk_inp, t, name="trunk_fourier")

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
        p, activation=None,
        kernel_regularizer=keras.regularizers.l2(l2_strength),
        name="branch_out",
    )(b)                                             # (B, p)  linear

    # ── Trunk evaluation inside the graph ─────────────────────────────
    T = layers.Lambda(
        lambda b_dummy: trunk_model(phi_const, training=False),
        name="trunk_eval",
    )(b)                                             # (n_z, p)

    # ── Combine: (B,p) x (n_z,p)^T -> (B, n_z)  then softplus ───────
    field = layers.Lambda(
        lambda inputs: tf.einsum("bp,jp->bj", inputs[0], inputs[1]),
        name="deeponet_field",
    )([b, T])                                        # (B, n_z)  linear

    out = layers.Activation("softplus", name="z_softplus")(field)

    return keras.Model(branch_inp, out, name="DeepONet_MLP_Fourier")


# ══════════════════════════════════════════════════════════════════════
# Compile helper  (In[19])
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
# Error metrics  (In[22])
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
    Z  = np.loadtxt(os.path.join(DATA_DIR, z_file), delimiter=",")
    Gz = np.loadtxt(os.path.join(DATA_DIR, d_file), delimiter=",")

    z_mean = np.mean(Z, axis=0)

    # ── Scaling ───────────────────────────────────────────────────────
    # Scalers are fit on this prior's data only.
    # scaler_G : normalises observations  (NN input)
    # scaler_z : normalises parameters    (NN output, in scaled space
    #            softplus acts as smooth non-negativity regularisation)
    # FIX: store scaled targets in Z_scaled not z_scaled to avoid
    # overwriting in the prediction loop below.
    scaler_G  = StandardScaler()
    Gz_scaled = scaler_G.fit_transform(Gz)               # (N, 15)

    scaler_z = StandardScaler()
    Z_scaled  = scaler_z.fit_transform(Z)                # (N, 100)

    d_scaled = scaler_G.transform(d_obs.reshape(1, -1))  # (1, 15)

    # ── Train / val split ─────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        Gz_scaled, Z_scaled,
        test_size=0.15,
        random_state=42,
    )

    input_dim  = X_train.shape[1]   # 15
    output_dim = y_train.shape[1]   # 100

    # ── Build models (fresh weights each prior) ───────────────────────
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
        # FIX: use z_pred_scaled / z_pred_unscaled, never reuse Z_scaled.
        z_pred_scaled   = model.predict(d_scaled, verbose=0)         # (1,100)
        z_pred_unscaled = scaler_z.inverse_transform(z_pred_scaled).ravel()

        all_predictions[prior_name][model_name] = z_pred_unscaled
        all_metrics[prior_name][model_name] = {
            "RMSE":   rmse_fn(z_pred_unscaled, z_true),
            "Rel_L2": rel_l2(z_pred_unscaled, z_true),
        }
        all_histories[prior_name][model_name] = {
            "loss":     history.history["loss"],
            "val_loss": history.history["val_loss"],
        }

        print(
            f"    RMSE   = {all_metrics[prior_name][model_name]['RMSE']:.4f}  "
            f"Rel_L2 = {all_metrics[prior_name][model_name]['Rel_L2']:.4f}"
        )

    # ── Baselines ─────────────────────────────────────────────────────
    all_predictions[prior_name]["Gauss-Newton"] = z_GN
    all_predictions[prior_name]["Sample_Mean"]  = z_mean

    all_metrics[prior_name]["Gauss-Newton"] = {
        "RMSE":   rmse_fn(z_GN,   z_true),
        "Rel_L2": rel_l2(z_GN,    z_true),
    }
    all_metrics[prior_name]["Sample_Mean"] = {
        "RMSE":   rmse_fn(z_mean, z_true),
        "Rel_L2": rel_l2(z_mean,  z_true),
    }


# ══════════════════════════════════════════════════════════════════════
# Save CSVs — schema mirrors wing convention for analysis compatibility
# ══════════════════════════════════════════════════════════════════════

# ── gravity_predictions.csv ───────────────────────────────────────────
rows_predictions = []
for prior_name, model_dict in all_predictions.items():
    for model_name, z_pred in model_dict.items():
        for i, val in enumerate(z_pred):
            rows_predictions.append({
                "w_index":    i,
                "w":          w[i],
                "prior":      prior_name,
                "model":      model_name,
                "prediction": float(val),
            })
    for i, val in enumerate(z_true):
        rows_predictions.append({
            "w_index":    i,
            "w":          w[i],
            "prior":      prior_name,
            "model":      "True",
            "prediction": float(val),
        })

pd.DataFrame(rows_predictions).to_csv(
    os.path.join(RESULTS_DIR, "gravity_predictions.csv"), index=False
)

# ── gravity_metrics.csv ───────────────────────────────────────────────
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
    os.path.join(RESULTS_DIR, "gravity_metrics.csv"), index=False
)

# ── gravity_histories.csv ─────────────────────────────────────────────
# Plain MSE training: loss and val_loss only (no physics loss term).
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
    os.path.join(RESULTS_DIR, "gravity_histories.csv"), index=False
)

print("\nSaved files:")
print(os.path.join(RESULTS_DIR, "gravity_predictions.csv"))
print(os.path.join(RESULTS_DIR, "gravity_metrics.csv"))
print(os.path.join(RESULTS_DIR, "gravity_histories.csv"))

# ── Console summary ───────────────────────────────────────────────────
print(f"\n{'Prior':<12} {'Model':<25} {'RMSE':>8} {'Rel_L2':>8}")
print("-" * 57)
for prior_name, model_dict in all_metrics.items():
    for model_name, m in model_dict.items():
        print(
            f"{prior_name:<12} {model_name:<25} "
            f"{m['RMSE']:>8.4f} {m['Rel_L2']:>8.4f}"
        )