import h5py, os
import numpy as np
import argparse
import tensorflow
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling1D
from qkeras import *
from tensorflow.keras.regularizers import l1
from sklearn.utils.class_weight import compute_class_weight

from inputFixer import add_ip
from plotting.kinematics_plotter import kinematics



from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow.keras.utils import plot_model

import tensorflow_model_optimization as tfmot
from sklearn.preprocessing import MinMaxScaler

from argparse import Namespace

N_FEAT = 13
N_PART_PER_JET = 10
tag = "train"
llpSample = "4b_4c_4u"

# Loaded once in main(), reused across all sweep trials
_data = {}

SWEEP_CONFIG = Namespace( batch_size = 32,
dense_units = 8,
#epochs = 300,
epochs = 1,
filters = 16,
l1_reg = 0.0000339421891017366,
learning_rate = 0.002559574085015032,
lr_decay_factor = 0.5,
lr_patience = 10,
momentum = 0.95,
n_conv_layers = 2,
n_dense_layers = 2,
sparsity = 0.75,
sparsity_begin_step = 3000,
sparsity_frequency = 50,
total_bits = 12 )

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
    
    for i in range(config.n_dense_layers):

        x = QDense(
            dense_units,
            kernel_quantizer=q_kernel, bias_quantizer=q_kernel,
            kernel_initializer="lecun_uniform",
            kernel_regularizer=l1(reg), bias_regularizer=l1(reg),
            name=f"q_dense_{i}",
        )(x)

        x = QActivation(activation=q_relu, name=f"q_activation_dense_{i}")(x)

    outputs = QDense(
        1,
        kernel_quantizer=q_kernel, bias_quantizer=q_kernel,
        kernel_initializer="lecun_uniform",
        kernel_regularizer=l1(reg), bias_regularizer=l1(reg),
        name="q_dense_o",
    )(x)
    #x = QActivation(activation="smooth_sigmoid", name="s_sigmoid")(x)
    #outputs = QActivation(activation=quantized_bits(tb, 1, alpha=1, keep_negative=False), name="q_sigmoid")(x)
    #outputs = Activation("sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs, name="model")

    plot_model(model, to_file=os.getcwd() + f"/{tag}_13_model.png", show_shapes=True, show_layer_names=True )

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
        weighted_metrics=[tensorflow.keras.metrics.AUC(name="auc")]
    )
    return model


def train_sweep(config):

    model = build_model(config)

    model.summary()

    

    callbacks = [
        tensorflow.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, verbose=1),
        pruning_callbacks.UpdatePruningStep(),
        tensorflow.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.lr_decay_factor,
            patience=config.lr_patience,
            verbose=1,
        ),
    ]

    history=model.fit(
        _data["X"], _data["y"],
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=2,
        sample_weight=np.asarray(_data["weights"]),
        class_weight = _data["class_Weights"],
        validation_split=0.20,
        callbacks=callbacks,
    )

    # Log actual sparsity of each prunable layer after training
    stripped = tfmot.sparsity.keras.strip_pruning(model)

    print("Training finished. Keys of history object: " + str(history.history.keys()))


    plt.figure(figsize=(7,5), dpi=120)
    plt.plot(history.history['loss'], label = 'Train')
    plt.plot(history.history['val_loss'], label = 'Validation')
    plt.title('Model Loss', fontsize=25)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.tight_layout()
    #plt.savefig(str(args.llpType) + "/qkLoss{}.pdf".format(str(args.llpType)) , dpi=120 )
    plt.savefig(os.getcwd() + "/{}_epochs_{epochs}_13qkLoss_LossFromLogits.pdf".format(tag, epochs=history.epoch[-1]) , dpi=120 )

    plt.figure(figsize=(7,5), dpi=120)
    plt.plot(history.history['binary_accuracy'], label = 'Train')
    plt.plot(history.history['val_binary_accuracy'], label = 'Validation')
    plt.title('Model Accuracy', fontsize=25)
    plt.ylabel('BinaryAccuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.tight_layout()
    #plt.savefig(str(args.llpType) + "/qkLoss{}.pdf".format(str(args.llpType)) , dpi=120 )
    plt.savefig(os.getcwd() + "/{}_epochs_{epochs}_13qkAccuracy_LossFromLogits.pdf".format(tag, epochs=history.epoch[-1]) , dpi=120 )

    plt.figure(figsize=(7,5), dpi=120)
    plt.plot(history.history['lr'], label = 'Train')
    #plt.plot(history.history['val_binary_accuracy'], label = 'Validation')
    plt.title('Model Accuracy', fontsize=25)
    plt.ylabel('BinaryAccuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.tight_layout()
    #plt.savefig(str(args.llpType) + "/qkLoss{}.pdf".format(str(args.llpType)) , dpi=120 )
    plt.savefig(os.getcwd() + "/{}_epochs_{epochs}_13qkLR_LossFromLogits.pdf".format(tag, epochs=history.epoch[-1]) , dpi=120 )

    model = tfmot.sparsity.keras.strip_pruning(model)
    model.save( os.getcwd() + "/{}_epochs_{epochs}_13qkL1JetTagModel_LossFromLogits.h5".format(tag, epochs=history.epoch[-1]) )


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

    old_N_FEAT = 14
    X = dataset[:, 0 : len(dataset[0]) - 1]
    y = dataset[:, len(dataset[0]) - 1]
    X = X.reshape((X.shape[0], N_PART_PER_JET, old_N_FEAT)) #STILL 14 feats

    #change inputs to 13 feats
    X = add_ip(X) #add ip as feature instead of separate dx and dy features

    # select jets with pT > 1 GeV
    mask = sampleData[:, 0] > 20
    sampleData = sampleData[mask]
    y = y[mask]
    X = X[mask]

    kinematics(X, sampleData, y, llpSample, tag)


    print("====================================================")
    print("Number of bkg jets: ", len(X[y==0]))
    print("Number of signal jets: ", len(X[y==1]))
    print("====================================================")


    thebins = np.linspace(min(np.log(sampleData[:, 0])), max(np.log(sampleData[:, 0])), 101) # check for right range
    bkgPts = np.log(sampleData[y==0][:,0])
    sigPts = np.log(sampleData[y==1][:,0])
    bkg_counts, _ = np.histogram(bkgPts, bins=thebins)
    sig_counts, _ = np.histogram(sigPts, bins=thebins)
    total_bkg = len(bkgPts)
    total_sig  = len(sigPts)
    weights_pt = np.nan_to_num((sig_counts + 0.5) / (bkg_counts + 0.5), nan=total_sig / total_bkg)

    weights     = np.ones(len(y))
    pt_indicies = np.clip(np.digitize(np.log(sampleData[:, 0]), bins=thebins) - 1, 0, len(weights_pt) - 1)
    weights[y == 1] = weights_pt[pt_indicies][y == 1]

    #compute class weights
    classes = np.unique(y)
    cl_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, cl_weights)) #required by keras

       #Plot for understanding weights

    fig, (ax_main, ax_ratio) = plt.subplots(
        2, 1, figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )
    fig.subplots_adjust(hspace=0.05)

    # Main panel
    ax_main.step(thebins[:-1], sig_counts, where="post", label="Signal")
    ax_main.step(thebins[:-1], bkg_counts, where="post", label="Bkg")
    ax_main.set_ylabel("Events")
    ax_main.legend()

    # Ratio panel
    ax_ratio.step(thebins[:-1], weights_pt, where="post", color="black")
    ax_ratio.axhline(1.0, color="red", linestyle="--", linewidth=1)
    ax_ratio.set_ylabel("weights")
    ax_ratio.set_xlabel(r"$P_T$")
    fig.savefig(os.getcwd() + f"/{tag}_pt_weights.png", bbox_inches="tight", dpi=150 )

    _data["X"]       = X
    _data["y"]       = y
    _data["weights"] = weights
    _data["class_Weights"] = class_weight_dict

    kinematics(_data["X"], sampleData, _data["y"], "4b_4c_4u", "train" )

    #call training
    train_sweep(SWEEP_CONFIG)






#=============================================

def sanity_check():
    return None




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments")

    parser.add_argument("--sanity", action="store_true", help="Do sanity checks.")
    parser.add_argument("SignalTrainFile",       type=str)
    parser.add_argument("BkgTrainFile",          type=str)
    parser.add_argument("sig_jetData_TrainFile", type=str)
    parser.add_argument("bkg_jetData_TrainFile", type=str)
    args = parser.parse_args()

    if args.sanity:
        sanity_check()
    else:
        main(args)