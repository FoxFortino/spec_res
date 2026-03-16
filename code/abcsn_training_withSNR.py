import sys
import os
import argparse
import shutil

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy import stats

import tensorflow as tf
import keras
import keras_hub

from keras.layers import Dense
from keras.layers import ReLU
from keras.layers import Reshape
from keras.layers import UpSampling2D
from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import GlobalMaxPooling1D
from keras.layers import Concatenate
from keras.layers import MultiHeadAttention
from keras.layers import Masking
from keras.regularizers import L1L2

from keras.optimizers import Nadam
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.metrics import F1Score

from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint

from keras_hub.layers import SinePositionEncoding
from keras_hub.layers import TransformerEncoder

# My packages
import abcsn_config
import data_preparation as dp
import data_plotting as dplt

from icecream import ic


rng = np.random.RandomState(1415)
keras.utils.set_random_seed(1415)
tf.config.experimental.enable_op_determinism()


def ptf(
    model_name,
    mask_frac=0.15,
    freeze_enc=False,
    num_epochs_pretrain=10_000,
    num_epochs_transfer=10_000,
    batch_size_pretrain=64,
    batch_size_transfer=64,
    lr0_pretrain=1e-4,
    lr0_transfer=1e-5,
    patience_es_pretrain=25,
    patience_es_transfer=25,
    patience_rlrp_pretrain=10,
    patience_rlrp_transfer=10,
    factor_rlrp_pretrain=0.5,
    factor_rlrp_transfer=0.5,
    minlr_rlrp_pretrain=1e-7,
    minlr_rlrp_transfer=1e-7,
    mindelta_pretrain=0.0005,
    mindelta_transfer=0.005,
    PE="fourier",
    intermediate_dim=64,
    num_heads=8,
    do_enc=0.50,
    act_ff="leaky_relu",
    do_ff=0.50,
    l2=0.01,
    l1=0,
    save_dir="/home/2649/repos/SCS/nb/pre_training",
):
    R = 100
    model_dir = os.path.join(save_dir, model_name)
    ic(model_dir)
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)
    
    # Load data
    Xtrn, Xtst, Ytrn, Ytst, num_wvl, num_classes, sn_dict_trn, sn_dict_tst, wvl = load_data()
    
    # Mask the data
    mask_val = 0
    wvl_range = (4500, 7000)
    nonzero_bool = ~((wvl < wvl_range[0]) | (wvl > wvl_range[1]))
    Xtrn_masked = get_masked_spectrum(Xtrn, wvl, nonzero_bool, mask_frac, mask_val)
    Xtst_masked = get_masked_spectrum(Xtst, wvl, nonzero_bool, mask_frac, mask_val)
    
    # Make the masked self-attention model
    input_shape = Xtrn_masked.shape[1:]
    model_pretrain = make_model(
        input_shape,
        pretrain=True,
        PE=PE,
        intermediate_dim=intermediate_dim,
        num_heads=num_heads,
        do_enc=do_enc)
    model_pretrain.summary()
    
    # Initialize the masked self-attention model
    loss = keras.losses.MeanSquaredError(name="mse")
    optimizer = keras.optimizers.Nadam(learning_rate=lr0_pretrain)
    model_pretrain.compile(loss=loss, optimizer=optimizer)
    
    # Train the masked self-attention model
    callbacks_pretrain = get_callbacks(
        patience_es_pretrain,
        patience_rlrp_pretrain,
        factor_rlrp_pretrain,
        minlr_rlrp_pretrain,
        mindelta_pretrain,
        os.path.join(model_dir, "pretrain_log.csv"),
    )
    history_pretrain = model_pretrain.fit(
        Xtrn_masked,
        Xtrn,
        validation_data=(Xtst_masked, Xtst),
        epochs=num_epochs_pretrain,
        batch_size=batch_size_pretrain,
        verbose=2,
        callbacks=callbacks_pretrain,
    )

    model_pretrain.save(os.path.join(model_dir, "ABCSN_pretrain.keras"))
    
    # Gather the encoder layers of the pre-train model.
    layers_pretrain_encoder = []
    for layer in history_pretrain.model.layers:
        if "encoder" in layer.name:
            layers_pretrain_encoder.append(layer)

    visualize_model(
        R,
        model_dir,
        history_pretrain,
        callbacks_pretrain[0],
        patience_es_pretrain,
        pretrain=True,
        X=Xtst,
        X_masked=Xtst_masked,
        wvl=wvl,
        fig_masked=True,
        fig_pretrain_loss=True)
    
    
    # Make the new model
    model_transfer = make_model(
        input_shape,
        pretrain=False,
        num_classes=num_classes,
        PE=PE,
        intermediate_dim=intermediate_dim,
        num_heads=num_heads,
        do_enc=do_enc,
        act_ff=act_ff,
        do_ff=do_ff,
        l2=l2,
        l1=l1)
    
    # Loop through each layer in the new model
    # If it's an encoder layer, it will be frozen and also gathered into a list.
    layers_transfer_encoder = []
    for i, layer in enumerate(model_transfer.layers):
        if "encoder" in layer.name:
            layers_transfer_encoder.append(layer)
            if freeze_enc:
                layer.trainable = False
        print(i, layer.trainable, layer)
    
    # Loop through the encoder layers in the pretrain and the transfer model.
    # Set the transfer model weights to the weights of the pretrain model.
    for layer_transfer, layer_pretrain in zip(layers_transfer_encoder, layers_pretrain_encoder):
        layer_transfer.set_weights(layer_pretrain.get_weights())
    
    # Check the summary to make sure the trainable and non-trainable weights are set.
    model_transfer.summary()
    
    # Initialize the transfer learning model.
    loss = keras.losses.CategoricalCrossentropy()
    acc = keras.metrics.CategoricalAccuracy(name="ca")
    f1 = keras.metrics.F1Score(average="macro", name="f1")
    optimizer = keras.optimizers.Nadam(learning_rate=lr0_transfer)
    model_transfer.compile(loss=loss, optimizer=optimizer, metrics=[acc, f1])
    
    # Train the transfer learning model.
    callbacks_transfer = get_callbacks(
        patience_es_transfer,
        patience_rlrp_transfer,
        factor_rlrp_transfer,
        minlr_rlrp_transfer,
        mindelta_transfer,
        os.path.join(model_dir, "transfer_log.csv"),
        monitor_es="val_f1",
        mode_es="max",
    )
    history_transfer = model_transfer.fit(
        Xtrn,
        Ytrn,
        validation_data=(Xtst, Ytst),
        epochs=num_epochs_transfer,
        batch_size=batch_size_transfer,
        verbose=2,
        callbacks=callbacks_transfer,
    )

    model_transfer.save(os.path.join(model_dir, "ABCSN.keras"))
        
    visualize_model(
        R,
        model_dir,
        history_transfer,
        callbacks_transfer[0],
        patience_es_transfer,
        transfer=True,
        Xtrn=Xtrn,
        Xtst=Xtst,
        Ytrn=Ytrn,
        Ytst=Ytst,
        num_classes=num_classes,
        sn_dict_trn=sn_dict_trn,
        sn_dict_tst=sn_dict_tst,
        fig_loss=True,
        fig_CMtst=True,
        fig_CMtrn=True,
        fig_cal=True,
    )
    return


def make_model(
    input_shape,
    pretrain=False,
    num_classes=None,
    PE=None,
    intermediate_dim=256,
    num_heads=8,
    do_enc=0.50,
    act_ff="relu",
    do_ff=0.50,
    l2=1e-5,
    l1=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs

    x = Dense(1024, activation="linear")(x)
    x = Reshape((64, 16))(x)
    x = Dense(128, activation='relu')(x)

    if PE == "RoPE":
        x = RotaryEmbedding()(x)
    elif PE == "vaswani":
        PE = SinePositionEncoding()(x)
        x = x + PE
    elif PE == "fourier":
        PE = SinePositionEncoding()(x)
        PE = Dense(64, activation="gelu")(PE)
        PE = Dense(x.shape[2], activation="linear")(PE)
        x = x + PE

    x = TransformerEncoder(
        intermediate_dim=intermediate_dim,
        num_heads=num_heads,
        dropout=do_enc,
        normalize_first=True)(x)
    x = TransformerEncoder(
        intermediate_dim=intermediate_dim,
        num_heads=num_heads,
        dropout=do_enc,
        normalize_first=True)(x)
    x = TransformerEncoder(
        intermediate_dim=intermediate_dim,
        num_heads=num_heads,
        dropout=do_enc,
        normalize_first=True)(x)
    x = TransformerEncoder(
        intermediate_dim=intermediate_dim,
        num_heads=num_heads,
        dropout=do_enc,
        normalize_first=True)(x)
    x = TransformerEncoder(
        intermediate_dim=intermediate_dim,
        num_heads=num_heads,
        dropout=do_enc,
        normalize_first=True)(x)
    x = TransformerEncoder(
        intermediate_dim=intermediate_dim,
        num_heads=num_heads,
        dropout=do_enc,
        normalize_first=True)(x)

    if pretrain:
        x = Reshape((1, x.shape[1] * x.shape[2]))(x)
        outputs = Dense(input_shape[1], activation="linear", use_bias=False)(x)

    else:
        x = Flatten()(x)
        x = Dense(
            1024,
            activation=act_ff,
            kernel_regularizer=L1L2(l2=l2, l1=l1))(x)
        x = Dropout(do_ff)(x)
        x = Dense(
            256,
            activation=act_ff,
            kernel_regularizer=L1L2(l2=l2, l1=l1))(x)
        x = Dropout(do_ff)(x)
        x = Dense(
            64,
            activation=act_ff,
            kernel_regularizer=L1L2(l2=l2, l1=l1))(x)
        x = Dropout(do_ff)(x)
        outputs = Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs)

    return model


def load_data():
    # Load pre-prepared data.
    df_trn = pd.read_parquet("../data/resolution_100_parquet/df_SPAR_trn.parquet")
    df_tst = pd.read_parquet("../data/resolution_100_parquet/df_SPR_tst.parquet")

    # df_trn = pd.read_parquet("/home/2649/repos/SCS/data/R100/df_PAR_trn.parquet")
    # df_tst = pd.read_parquet("/home/2649/repos/SCS/data/R100/df_PR_tst.parquet")

    # Extract the data from the files.
    Xtrn, Ytrn, num_trn, num_wvl, num_classes, sn_dict_trn, wvl = dp.extract(df_trn, return_wvl=True)
    Xtst, Ytst, num_tst, num_wvl, num_classes, sn_dict_tst, wvl = dp.extract(df_tst, return_wvl=True)

    # Shuffle the data (This is redundant because Keras should be shuffling anyway.)
    trn_shuffle_inds = rng.choice(num_trn, size=num_trn, replace=False)
    tst_shuffle_inds = rng.choice(num_tst, size=num_tst, replace=False)
    Xtrn = Xtrn[trn_shuffle_inds, None, :]
    Xtst = Xtst[tst_shuffle_inds, None, :]
    Ytrn = Ytrn[trn_shuffle_inds, :]
    Ytst = Ytst[tst_shuffle_inds, :]

    return Xtrn, Xtst, Ytrn, Ytst, num_wvl, num_classes, sn_dict_trn, sn_dict_tst, wvl


def get_masked_spectrum(spectrum, wvl, nonzero_bool, mask_frac, mask_val):
    spectrum_masked = spectrum.copy()  # Making this a copy is really really important to do, don't forget it!
    nonzero_spectrum_masked = spectrum_masked[..., nonzero_bool]
    num_maskable_bins = nonzero_spectrum_masked.shape[-1]
    # ic(num_maskable_bins)

    # Calculate how many bins corresponds to mask_frac
    bins_masked = np.ceil(num_maskable_bins * mask_frac).astype(int)
    # ic(bins_masked)

    # We are going to mask a contiguous set of `bin_masked` bins in the spectrum.
    # The following code figures out where that set will be.
    mask_location = stats.randint.rvs(0, num_maskable_bins, loc=0, size=1, random_state=rng)[0]
    # ic(mask_location)

    # Now we want to get `bins_masked` indices centered on `mask_location`.
    first_bin = (mask_location - np.floor(bins_masked / 2)).astype(int)
    last_bin = (mask_location + np.ceil(bins_masked / 2)).astype(int)
    masked_bins = np.array([i for i in range(first_bin, last_bin)])
    # ic(first_bin, last_bin)
    # ic(masked_bins)

    # However, if `mask_location` is on the edge of the spectrum then we needto shift the masked bins so that `bins_masked` number of bins are always masked even if `mask_location` is on the edge.
    if first_bin < 0:
        masked_bins -= first_bin
    elif last_bin > num_maskable_bins:
        masked_bins -= last_bin - num_maskable_bins
    # ic(masked_bins)

    # Officially mask the values on `nonzero_spectrum_masked` which does not have any zero padding at the beginning or end.
    nonzero_spectrum_masked[..., masked_bins] = mask_val

    # Perturb 2.5% of the flux values.
    num_bins_perturbed = np.ceil(num_maskable_bins * 0.025).astype(int)
    # ic(num_bins_perturbed)

    spectrum_indices = np.arange(num_maskable_bins)
    perturbable_bins = np.array(list(set(spectrum_indices) - set(masked_bins)))
    # ic(perturbable_bins)

    perturbed_bins = rng.choice(perturbable_bins, size=num_bins_perturbed)
    # ic(perturbed_bins)

    perturbations = np.abs(stats.norm.rvs(loc=0, scale=1, size=num_bins_perturbed, random_state=rng))
    # ic(perturbations)

    nonzero_spectrum_masked[..., perturbed_bins] *= perturbations

    # Finally put the masked values into `spectrum_masked` which has thezero padding at the start and end.
    spectrum_masked[..., nonzero_bool] = nonzero_spectrum_masked
    return spectrum_masked
get_masked_spectrum = np.vectorize(get_masked_spectrum, signature='(n),(n),(n),(),()->(n)')


def get_callbacks(
    patience_es,
    patience_rlrp,
    factor_rlrp,
    minlr_rlrp,
    mindelta,
    logfile,
    monitor_es="val_loss",
    mode_es="min",
    monitor_rlrp="val_loss",
    mode_rlrp="min",
):
    cb_es = keras.callbacks.EarlyStopping(
        monitor=monitor_es,
        min_delta=mindelta,
        patience=patience_es,
        verbose=2,
        mode=mode_es,
        restore_best_weights=True,
    )

    cb_rlrp = keras.callbacks.ReduceLROnPlateau(
        monitor=monitor_rlrp,
        factor=factor_rlrp,
        patience=patience_rlrp,
        verbose=2,
        mode=mode_rlrp,
        min_delta=mindelta,
        cooldown=0,
        min_lr=minlr_rlrp,
    )

    cb_log = keras.callbacks.CSVLogger(
        logfile,
        separator=",",
        append=False,
    )


    return [cb_es, cb_rlrp, cb_log]


def visualize_model(
    R,
    model_dir,
    history,
    cb_es,
    patience_es,
    pretrain=False,
    transfer=False,
    finetune=False,
    X=None,
    X_masked=None,
    wvl=None,
    Xtrn=None,
    Xtst=None,
    Ytrn=None,
    Ytst=None,
    num_classes=None,
    sn_dict_trn=None,
    sn_dict_tst=None,
    fig_masked=False,
    fig_pretrain_loss=False,
    fig_loss=False,
    fig_CMtst=False,
    fig_CMtrn=False,
    fig_cal=False,
):
    # Take a look at how the reconstructed spectra look
    if fig_masked:
        fig = plot_masked_predictions(X, X_masked, wvl, history.model)
        fig.savefig(os.path.join(model_dir, f"fig00_pretrain_masked_spectra_reconstruction.png"))

    # Plot pre-training loss curve
    if fig_pretrain_loss:
        fig = plot_loss(history.history, cb_es=cb_es, patience_es=patience_es)
        fig.show()
        fig.savefig(os.path.join(model_dir, "fig01_pretrain_loss.png"))

    # Plot loss (transfer or finetune models)
    if fig_loss:
        history.history["epoch"] = np.arange(len(history.history["loss"]))
        fig = dplt.plot_loss(history.history, scale=6, cb_es=cb_es, patience_es=patience_es)
        fig.show()
        if transfer:
            fig.savefig(os.path.join(model_dir, "fig02_transfer_loss.png"))
        elif finetune:
            fig.savefig(os.path.join(model_dir, "fig06_finetune_loss.png"))

    # Evaluate the model to calculate final metrics.
    if fig_CMtrn:
        loss_trn, ca_trn, f1_trn = history.model.evaluate(x=Xtrn, y=Ytrn, verbose=0)
        Ptrn = history.model.predict(Xtrn, verbose=0)
        Ytrn_argmax = np.argmax(Ytrn, axis=1)
        Ptrn_argmax = np.argmax(Ptrn, axis=1)

        # Get the labels for the confusion matrix
        sn_dict_trn_inv = {v: k for k, v in sn_dict_trn.items()}
        SNtypes_int = np.unique(Ytrn_argmax)
        SNtypes_str = [abcsn_config.SN_Stypes_int_to_str[sn_dict_trn_inv[sn]] for sn in SNtypes_int]
    
        CMtrn = confusion_matrix(Ytrn_argmax, Ptrn_argmax)
        fig_title = f"Train Set | R = {R} | CA = {ca_trn:.4f} | F1 = {f1_trn:.4f}"
        fig = dplt.plot_cm(CMtrn, SNtypes_str, R, title=fig_title)
        fig.tight_layout()
        fig.show()
        if transfer:
            fig.savefig(os.path.join(model_dir, "fig03_transfer_CMtrn.png"))
        elif finetune:
            fig.savefig(os.path.join(model_dir, "fig07_finetune_CMtrn.png"))

    if fig_CMtst:
        loss_tst, ca_tst, f1_tst = history.model.evaluate(x=Xtst, y=Ytst, verbose=0)
        Ptst = history.model.predict(Xtst, verbose=0)
        Ytst_argmax = np.argmax(Ytst, axis=1)
        Ptst_argmax = np.argmax(Ptst, axis=1)

        # Get the labels for the confusion matrix
        sn_dict_tst_inv = {v: k for k, v in sn_dict_tst.items()}
        SNtypes_int = np.unique(Ytrn_argmax)
        SNtypes_str = [abcsn_config.SN_Stypes_int_to_str[sn_dict_trn_inv[sn]] for sn in SNtypes_int]

        # Plot confusion matrices for test and train sets.
        CMtst = confusion_matrix(Ytst_argmax, Ptst_argmax)
        fig_title = f"Test Set | R = {R} | CA = {ca_tst:.4f} | F1 = {f1_tst:.4f}"
        fig = dplt.plot_cm(CMtst, SNtypes_str, R, title=fig_title)
        fig.tight_layout()
        fig.show()
        if transfer:
            fig.savefig(os.path.join(model_dir, "fig04_transfer_CMtst.png"))
        elif finetune:
            fig.savefig(os.path.join(model_dir, "fig08_finetune_CMtst.png"))

    if fig_cal:
        Ptst = history.model.predict(Xtst, verbose=0)
        Ytst_argmax = np.argmax(Ytst, axis=1)
        Ptst_argmax = np.argmax(Ptst, axis=1)
        
        sn_dict_tst_inv = {v: k for k, v in sn_dict_tst.items()}
        SNtypes_int = np.unique(Ytrn_argmax)
        SNtypes_str = [abcsn_config.SN_Stypes_int_to_str[sn_dict_trn_inv[sn]] for sn in SNtypes_int]
        
        fig = plot_calibration(num_classes, Ptst, Ytst, Ytst_argmax, SNtypes_str)
        fig.show()
        if transfer:
            fig.savefig(os.path.join(model_dir, "fig05_transfer_calibration.png"))
        elif finetune:
            fig.savefig(os.path.join(model_dir, "fig09_finetune_calibration.png"))

    return


def plot_masked_predictions(X, X_masked, wvl, model):
    P = model.predict(X)

    fig, axes = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=False, figsize=(14, 7))
    fig.subplots_adjust(hspace=0)

    for ax in axes.flatten():
        i = np.random.choice(X.shape[0], size=1)[0]
        ax.plot(wvl, X[i, 0, :], c="tab:blue", ls="-", marker="o", markersize=3, label=f"Spectrum\n(index {i})")
        ax.plot(wvl, X_masked[i, 0, :], c="tab:blue", ls=":", marker="o", markersize=3, label="Masked")
        ax.plot(wvl, P[i, 0, :], c="tab:orange", ls="-", marker="o", markersize=3, label="Predicted")
        ax.legend()
        ax.set_xlim((4250, 7250))

    return fig


def plot_loss(log, scale=4, cb_es=None, patience_es=None):
    figsize = (scale * np.sqrt(2), scale)
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=False, figsize=figsize)
    fig.subplots_adjust(hspace=0)
    ax.plot(log["loss"], c="tab:blue", label="Training")
    ax.plot(log["val_loss"], c="tab:orange", label="Validation")
    if cb_es is not None:
        best_epoch = cb_es.stopped_epoch - (patience_es - 1)
        ax.axvline(x=best_epoch, c="k", ls=":", label=f"Saved model at epoch {best_epoch}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_xlim((0, None))
    ax.legend(loc="upper right")
    ax.grid()
    ax.set_yscale("log")
    fig.tight_layout()
    return fig


def plot_calibration(num_classes, P, Y, Y_argmax, SNtypes_str):
    # Since our data has 10 classes, we will make 10 calibration plots; one for each class.
    # Each calibrarion plot will be as if we are considering a binary classification problem (one vs rest).
    fig, axes = plt.subplots(
        nrows=num_classes, sharex=False, sharey=True, figsize=(8, 40)
    )
    fig.subplots_adjust(hspace=0.3)

    for sntype in range(num_classes):
        pred_prob_sntype = P[:, sntype]
        true_vals_sntype = Y[:, sntype]

        hist, bin_edges = np.histogram(pred_prob_sntype, bins=10, range=(0, 1))
        digitized = np.digitize(pred_prob_sntype, bin_edges, right=True)

        average_pred = []
        average_true = []
        stddev_pred = []
        stddev_true = []
        bins = np.unique(digitized)
        for bin_number in bins:
            bin_index = digitized == bin_number

            average_pred.append(np.mean(pred_prob_sntype[bin_index]))
            average_true.append(np.mean(true_vals_sntype[bin_index]))

            stddev_pred.append(np.std(pred_prob_sntype[bin_index]))
            stddev_true.append(np.std(true_vals_sntype[bin_index]))

        axes[sntype].errorbar(
            average_pred,
            average_true,
            xerr=stddev_pred,
            yerr=stddev_true,
            elinewidth=1 / 2,
            capsize=5 / 2,
            marker="o",
            ls="-",
            lw=0.5,
            label=f"{SNtypes_str[sntype]} ({(Y_argmax == sntype).sum()})",
        )
        axes[sntype].plot([0, 1], [0, 1], linestyle="--", lw=1)
        axes[sntype].set_ylabel("Average True Label")
        axes[sntype].set_ylim((0, 1))
        axes[sntype].set_xlim((0, 1))
        axes[sntype].grid()
        axes[sntype].legend(loc="upper left")

        axes[sntype].set_xlabel("Average Predicted Probability")

    fig.tight_layout()
    return fig


# def visualize_self_attention_weights(encoder_layers, scale=4, xscale=1, yscale=1):
#     # Figure out how many rows and columns the figure should have
#     # The number of rows will be equal to the maximum number of attention heads across all encoder layers
#     # The number of columns will be equal to the number of encoder layers.
#     nrows = -1
#     weights = []
#     for i in encoder_layers:
#         weights_i = i.weights[6]
#         num_heads_i = weights_i.shape[0]
#         if num_heads_i > nrows:
#             nrows = num_heads_i
#         weights.append(weights_i)

#     ncols = len(encoder_layers)

#     figsize = np.array([nrows * xscale, ncols * yscale]) * scale
#     fig, axes = plt.subplots(
#         ncols=ncols, nrows=nrows, sharex=True, sharey=True, figsize=figsize
#     )

#     for i, layer_ax in enumerate(axes.T):
#         layer_ax[0].set_title(f"Encoder {i+1}", size=10 * scale)
#         for j, head_ax in enumerate(layer_ax):
#             head_ax.imshow(weights[i][j])
#             head_ax.tick_params(
#                 axis="y",
#                 which="both",
#                 bottom=False,
#                 labelbottom=False,
#                 top=False,
#                 labeltop=False,
#                 left=False,
#                 labelleft=False,
#                 right=False,
#                 labelright=False,
#             )
#         head_ax.tick_params(
#             axis="x", which="both", labelsize=10 * scale, width=scale, length=3 * scale
#         )

#     fig.tight_layout()
#     return fig


# def visualize_dense_weights(dense_weights, scale=5):
#     fig, ax = plt.subplots(figsize=(scale * np.sqrt(2), scale))
#     ax.imshow(dense_weights)
#     ax.set_title("Weights for final dense layer")
#     fig.tight_layout()
#     return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--mask_frac", type=float)
    parser.add_argument("--freeze_enc", action="store_true")
    parser.add_argument("--num_epochs_pretrain", type=int)
    parser.add_argument("--num_epochs_transfer", type=int)
    parser.add_argument("--batch_size_pretrain", type=int)
    parser.add_argument("--batch_size_transfer", type=int)
    parser.add_argument("--lr0_pretrain", type=float)
    parser.add_argument("--lr0_transfer", type=float)
    parser.add_argument("--patience_es_pretrain", type=int)
    parser.add_argument("--patience_es_transfer", type=int)
    parser.add_argument("--patience_rlrp_pretrain", type=int)
    parser.add_argument("--patience_rlrp_transfer", type=int)
    parser.add_argument("--factor_rlrp_pretrain", type=float)
    parser.add_argument("--factor_rlrp_transfer", type=float)
    parser.add_argument("--minlr_rlrp_pretrain", type=float)
    parser.add_argument("--minlr_rlrp_transfer", type=float)
    parser.add_argument("--mindelta_pretrain", type=float)
    parser.add_argument("--mindelta_transfer", type=float)
    parser.add_argument("--PE", type=str)
    parser.add_argument("--intermediate_dim", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--do_enc", type=float)
    parser.add_argument("--act_ff", type=str)
    parser.add_argument("--do_ff", type=float)
    parser.add_argument("--l2", type=float)
    parser.add_argument("--l1", type=float)
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()

    ic(args)
    
    ptf(
        args.model_name,
        mask_frac=args.mask_frac,
        freeze_enc=args.freeze_enc,
        num_epochs_pretrain=args.num_epochs_pretrain,
        num_epochs_transfer=args.num_epochs_transfer,
        batch_size_pretrain=args.batch_size_pretrain,
        batch_size_transfer=args.batch_size_transfer,
        lr0_pretrain=args.lr0_pretrain,
        lr0_transfer=args.lr0_transfer,
        patience_es_pretrain=args.patience_es_pretrain,
        patience_es_transfer=args.patience_es_transfer,
        patience_rlrp_pretrain=args.patience_rlrp_pretrain,
        patience_rlrp_transfer=args.patience_rlrp_transfer,
        factor_rlrp_pretrain=args.factor_rlrp_pretrain,
        factor_rlrp_transfer=args.factor_rlrp_transfer,
        minlr_rlrp_pretrain=args.minlr_rlrp_pretrain,
        minlr_rlrp_transfer=args.minlr_rlrp_transfer,
        mindelta_pretrain=args.mindelta_pretrain,
        mindelta_transfer=args.mindelta_transfer,
        PE=args.PE,
        intermediate_dim=args.intermediate_dim,
        num_heads=args.num_heads,
        do_enc=args.do_enc,
        act_ff=args.act_ff,
        do_ff=args.do_ff,
        l2=args.l2,
        l1=args.l1,
    )
    sys.exit(0)

