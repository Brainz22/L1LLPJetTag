import h5py, os
import numpy as np
import argparse
import tensorflow
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling1D
from qkeras import *
from tensorflow.keras.regularizers import l1
import wandb
from wandb.integration.keras import WandbMetricsLogger

from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow.keras.utils import plot_model

import tensorflow_model_optimization as tfmot
from sklearn.preprocessing import MinMaxScaler

N_FEAT = 14
N_PART_PER_JET = 10

# Loaded once in main(), reused across all sweep trials
_data = {}

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        # Architecture
        "filters":            {"values": [8, 10, 16, 32]},
        "dense_units":        {"values": [8, 10, 16, 32]},
        "n_conv_layers":      {"values": [1, 2, 3]},
        # Quantization — too few bits is a common cause of high loss
        "total_bits":         {"values": [6, 8, 10, 12]},
        # Regularization
        "l1_reg":             {"min": 1e-5, "max": 1e-2, "distribution": "log_uniform_values"},
        # Optimizer
        "learning_rate":      {"min": 1e-4, "max": 1e-2, "distribution": "log_uniform_values"},
        "momentum":           {"values": [0.85, 0.9, 0.95, 0.99]},  # Adam beta_1
        "batch_size":         {"values": [32, 64, 128, 256]},
        # LR schedule
        "lr_decay_factor":    {"values": [0.1, 0.2, 0.5]},
        "lr_patience":        {"values": [10, 15, 25]},
        # Pruning / sparsity
        "sparsity":           {"values": [0.5, 0.75, 0.9]},
        "sparsity_begin_step":{"values": [500, 1000, 2000, 3000]},
        "sparsity_frequency": {"values": [50, 100, 200]},
        "epochs":             {"values": [100, 200, 300, 500]},
    },
}


def build_model(config):
    filters     = config.filters
    dense_units = config.dense_units
    reg         = config.l1_reg
    tb          = config.total_bits
    ib          = tb // 2  # integer bits always half of total

    q_kernel = quantized_bits(tb, ib, alpha=1)
    q_relu   = quantized_relu(tb, ib)

    x = inputs = Input(shape=(N_PART_PER_JET, N_FEAT), name="input_1")
    x = QActivation(activation=quantized_bits(12, 6, alpha=1), name="q_input")(x)

    for i in range(config.n_conv_layers):
        x = QConv1D(
            filters=filters, kernel_size=1, strides=1,
            kernel_quantizer=q_kernel, bias_quantizer=q_kernel,
            kernel_initializer="lecun_uniform",
            kernel_regularizer=l1(reg), bias_regularizer=l1(reg),
            name=f"q_conv1d_{i}",
        )(x)
        x = QActivation(activation=q_relu, name=f"q_activation_{i}")(x)

    x = GlobalAveragePooling1D(name="global_average_pooling1d")(x)

    x = QDense(
        dense_units,
        kernel_quantizer=q_kernel, bias_quantizer=q_kernel,
        kernel_initializer="lecun_uniform",
        kernel_regularizer=l1(reg), bias_regularizer=l1(reg),
        name="q_dense",
    )(x)
    x = QActivation(activation=q_relu, name="q_activation_dense")(x)

    outputs = QDense(
        1,
        kernel_quantizer=q_kernel, bias_quantizer=q_kernel,
        kernel_initializer="lecun_uniform",
        kernel_regularizer=l1(reg), bias_regularizer=l1(reg),
        name="q_dense_1",
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name="model")

    model = prune.prune_low_magnitude(
        model,
        pruning_schedule=pruning_schedule.ConstantSparsity(
            config.sparsity,
            begin_step=config.sparsity_begin_step,
            frequency=config.sparsity_frequency,
        ),
    )

    model.compile(
        loss=tensorflow.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tensorflow.keras.optimizers.Adam(
            learning_rate=config.learning_rate,
            beta_1=config.momentum,
        ),
        metrics=["binary_accuracy"],
    )
    return model


def train_sweep():
    with wandb.init() as run:
        config = wandb.config

        model = build_model(config)

        callbacks = [
            tensorflow.keras.callbacks.EarlyStopping(monitor="val_loss", patience=65, verbose=1),
            pruning_callbacks.UpdatePruningStep(),
            tensorflow.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=config.lr_decay_factor,
                patience=config.lr_patience,
                verbose=1,
            ),
            WandbMetricsLogger(),
        ]

        model.fit(
            _data["X"], _data["y"],
            epochs=config.epochs,
            batch_size=config.batch_size,
            verbose=2,
            sample_weight=np.asarray(_data["weights"]),
            validation_split=0.20,
            callbacks=callbacks,
        )

        # Log actual sparsity of each prunable layer after training
        stripped = tfmot.sparsity.keras.strip_pruning(model)
        for layer in stripped.layers:
            for w in layer.get_weights():
                if w.ndim > 1:  # skip bias vectors
                    sparsity = float(np.mean(w == 0))
                    wandb.log({f"sparsity/{layer.name}": sparsity})

def main(args):
    print("Reading signal from " + args.SignalTrainFile)
    print("Reading background from " + args.BkgTrainFile)
    print("Reading signal jet data from " + args.sig_jetData_TrainFile)
    print("Reading background jet data from " + args.bkg_jetData_TrainFile)

    with h5py.File(args.SignalTrainFile, "r") as hf:
        dataset = hf["jet_constituents"][:]
    with h5py.File(args.BkgTrainFile, "r") as hf:
        datasetQCD = hf["jet_constituents"][:]
    with h5py.File(args.sig_jetData_TrainFile, "r") as hf:
        sampleData = hf["train_jet_data"][:]
    with h5py.File(args.bkg_jetData_TrainFile, "r") as hf:
        sampleDataQCD = hf["train_jet_data"][:]

    dataset    = np.concatenate((dataset, datasetQCD))
    sampleData = np.concatenate((sampleData, sampleDataQCD))
    fullData   = np.concatenate((dataset, sampleData), axis=1)
    np.random.shuffle(fullData)
    dataset    = fullData[:, 0:141]
    sampleData = fullData[:, 141:]

    X = dataset[:, 0 : len(dataset[0]) - 1]
    y = dataset[:, len(dataset[0]) - 1]
    X = X.reshape((X.shape[0], N_PART_PER_JET, N_FEAT))

    normalizeIPs = False
    norm_b4 = max(X[:, :, 8].ravel()) < 2.0
    if not norm_b4:
        print("\nImpact parameter was not normalized beforehand.\n")

    if norm_b4:
        tag = "separateNorm/sepNorm_train"
    elif normalizeIPs:
        tag = "b4train_Norm/Norm_b4train"
        scaler = MinMaxScaler(feature_range=(-1, 1))
        for idx in [8, 9, 10]:
            flat = scaler.fit_transform([[v] for v in X[:, :, idx].ravel()])
            X[:, :, idx] = flat.reshape(X[:, :, idx].shape)
    else:
        tag = "noNorm/noNorm_train"

    thebins    = np.linspace(0, max(sampleData[:, 0]), 60)
    bkgPts     = sampleData[y == 0][:, 0]
    sigPts     = sampleData[y == 1][:, 0]
    bkg_counts, _ = np.histogram(bkgPts, bins=thebins)
    sig_counts, _ = np.histogram(sigPts, bins=thebins)
    total_bkg  = len(bkgPts)
    total_sig  = len(sigPts)
    weights_pt = np.nan_to_num(sig_counts / bkg_counts, nan=total_sig / total_bkg)

    weights     = np.ones(len(y))
    pt_indicies = np.clip(np.digitize(sampleData[:, 0], bins=thebins) - 1, 0, len(weights_pt) - 1)
    weights[y == 0] = weights_pt[pt_indicies][y == 0]

    _data["X"]       = X
    _data["y"]       = y
    _data["weights"] = weights

    sweep_id = wandb.sweep(SWEEP_CONFIG, project="L1LLPJetTag")
    wandb.agent(sweep_id, function=train_sweep, count=args.n_trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments")
    parser.add_argument("SignalTrainFile",       type=str)
    parser.add_argument("BkgTrainFile",          type=str)
    parser.add_argument("sig_jetData_TrainFile", type=str)
    parser.add_argument("bkg_jetData_TrainFile", type=str)
    parser.add_argument("llpType",               type=str)
    parser.add_argument("--n_trials", type=int, default=30, help="Number of sweep trials")
    args = parser.parse_args()
    main(args)