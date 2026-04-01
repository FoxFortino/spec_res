import sys
from os.path import join, isfile

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import display, clear_output

import spectral_features as sf
import measure_signal as ms
import dataset_utils as du

from icecream import ic
from IPython import embed

FILE_DATASET = "../data/forSNR/closer_look.parquet"
SORT_COLS = ["SN Subtype ID", "SN Name", "Spectrum Phase", "Spectrum Cardinality"]
SNRinfo_columns = [
    "Denoising Parameter",
    "minima_i",
    "searchBlu",
    "searchRed",
    "useBlu",
    "useRed",
    "maxBlu",
    "maxRed",
    "noiseWindowBlu",
    "noiseWindowRed",
]


def review(i=0, subtype=None):
    df_dataset = pd.read_parquet(FILE_DATASET)
    df_dataset.sort_values(SORT_COLS, inplace=True)
    df_dataset.reset_index(inplace=True, drop=True)
    
    if subtype is not None:
        subtype_mask = df_dataset["SN Subtype"] == subtype
        df_dataset = df_dataset[subtype_mask]

    wvl, spectra, df_meta = du.unpack_dataset(df_dataset)
    num_spectra, num_wvl = spectra.shape
    
    options = reset_options()

    user_input = None
    while True:
        ###
        df_dataset = pd.read_parquet(FILE_DATASET)
        df_dataset.sort_values(SORT_COLS, inplace=True)
        df_dataset.reset_index(inplace=True, drop=True)
        
        if subtype is not None:
            subtype_mask = df_dataset["SN Subtype"] == subtype
            df_dataset = df_dataset[subtype_mask]
    
        wvl, spectra, df_meta = du.unpack_dataset(df_dataset)
        num_spectra, num_wvl = spectra.shape
        ###
        df_meta_i = df_meta.loc[i]
        spec_i = spectra[i]
        
        identifier = (
            f'{i} | '
            f'{df_meta.loc[i, "SN Subtype"]} | '
            f'{df_meta.loc[i, "SN Name"]} at '
            f'{df_meta.loc[i, "Spectrum Phase"]} '
            f'({df_meta.loc[i, "Spectrum Cardinality"]})'
        )
        
        # if not np.isnan(df_meta.loc[i, "Denoising Parameter"]):
        # print(f"Parameters previously saved for this spectrum.")
        print(df_dataset.loc[i, SNRinfo_columns])

        specsnr = ms.SpectrumSNR(
            df_meta.loc[i, "SN Name"],
            df_meta.loc[i, "SN Subtype"],
            df_meta.loc[i, "Spectrum Phase"],
            wvl,
            spec_i)

        assert df_dataset.loc[i, "Denoising Parameter"] != -999
        assert not np.isnan(df_dataset.loc[i, "Denoising Parameter"])

        if np.isnan(df_dataset.loc[i, "minima_i"]):
            minima_i = None
        else:
            minima_i = df_dataset.loc[i, "minima_i"]

        specsnr.summarize()
        specsnr.minmax_normalize()
        specsnr.set_spectral_feature()
        specsnr.denoise_gaussian(df_dataset.loc[i, "Denoising Parameter"])
        specsnr.find_spectral_line(
            feature_search_bounds=(df_dataset.loc[i, "searchBlu"], df_dataset.loc[i, "searchRed"]),
            minima_i=minima_i,
            plot=True)
        display(plt.gcf())
        specsnr.find_spectral_shoulders(
            blu_shoulder_nudge=int(df_dataset.loc[i, "maxBlu"]),
            red_shoulder_nudge=int(df_dataset.loc[i, "maxRed"]))
        specsnr.calc_pEW()
        specsnr.measure_feature_noise(
            noise_window_blu=df_dataset.loc[i, "noiseWindowBlu"],
            noise_window_red=df_dataset.loc[i, "noiseWindowRed"],
            useBlu=df_dataset.loc[i, "useBlu"],
            useRed=df_dataset.loc[i, "useRed"])
        specsnr.measure_SNR(plot=True)
        display(plt.gcf())
        # specsnr.execute_algorithm(df_dataset.loc[i], review=True)

        review_spectrum(
            df_dataset.loc[i],
            wvl,
            spec_i, 
            specsnr)
        display(plt.gcf())

        print(identifier)

        user_input = input()
        clear_output(wait=True)
        plt.close()
        plt.close()
        plt.close()

        print(identifier)

        # If the user simply hits enter then we increment i and save the
        # options to the SNRmetadata file.
        if user_input == "":
            options = reset_options()
            i += 1
        else:
            i, options = logic(i, user_input, options, df_dataset)
        
        df_dataset.to_parquet(FILE_DATASET)

    return df_dataset


def options_into_df(i, options, df):
    df.loc[i, "Denoising Parameter"] = options["sd"]
    df.loc[i, "minima_i"] = options["minima_i"]
    df.loc[i, "searchBlu"] = options["searchBlu"]
    df.loc[i, "searchRed"] = options["searchRed"]
    df.loc[i, "useBlu"] = options["useBlu"]
    df.loc[i, "useRed"] = options["useRed"]
    df.loc[i, "maxBlu"] = options["maxBlu"]
    df.loc[i, "maxRed"] = options["maxRed"]
    df.loc[i, "noiseWindowBlu"] = options["noiseWindowBlu"]
    df.loc[i, "noiseWindowRed"] = options["noiseWindowRed"]
    return


def logic(i, user_input, options, df_dataset):
    if (user_input.lower() == "p") or (user_input.lower() == "prev"):
        return i - 1, reset_options()
    
    elif (user_input.lower() == "n") or (user_input.lower() == "next"):
        return i + 1, reset_options()
    
    elif (user_input.lower()[0] == "a") or (user_input.lower()[:7] == "advance"):
        return i + int(user_input.split(" ")[1]), reset_options()
    
    elif user_input[:4] == "goto":
        return int(user_input.split(" ")[1]), reset_options()
    
    elif user_input.split(" ")[0] in options.keys():
        key = user_input.split(" ")[0]
        val = user_input.split(" ")[1]
        if key in ["sd", "searchBlu", "searchRed", "noiseWindowBlu", "noiseWindowRed"]:
            options[key] = float(val)
            
        elif key in ["maxBlu", "maxRed", "minima_i"]:
            options[key] = int(val)

        elif key in ["useBlu", "useRed"]:
            if (val.lower() == "true") or (val.lower() == "t"):
                options[key] = True
            elif (val.lower() == "false") or (val.lower() == "f"):
                options[key] = False
        
        options_into_df(i, options, df_dataset)
        return i, options

    elif (user_input.lower() == "exclude") or (user_input.lower() == "x"):
        options["sd"] = -999
        return i, options

    else:
        try:
            sd = float(user_input)
            options["sd"] = sd
            options_into_df(i, options, df_dataset)
            return i, options
        except ValueError:
            print(f"Invalid command.")
            return i, options


def reset_options():
    options = {
        "sd": 10,
        "minima_i": np.nan,
        "searchBlu": 500,
        "searchRed": 0,
        "useBlu": True,
        "useRed": True,
        "maxBlu": 0,
        "maxRed": 0,
        "noiseWindowBlu": 100,
        "noiseWindowRed": 100,
    }
    return options


def load_FFTdenoised_data(
    file_FFTdenoise_signal="../data/original_resolution_parquet/bianco_denoising/signal/preprocessed_signal_only.parquet",
    file_FFTdenoise_noise="../data/original_resolution_parquet/bianco_denoising/noise/preprocessed_noise_only.parquet",
):
    FFTdenoise_signal = pd.read_parquet(file_FFTdenoise_signal)
    FFTdenoise_noise = pd.read_parquet(file_FFTdenoise_noise)
    metadata_cols = [
        "SN Name",
        "SN Subtype",
        "SN Subtype ID",
        "SN Maintype",
        "SN Maintype ID",
        "Spectrum Phase",
        "Exclude",
        "Training Set",
    ]

    FFTdenoise_signal.reset_index(inplace=True)
    FFTd_S_metadata = FFTdenoise_signal[metadata_cols].copy(deep=True)
    FFTd_S_data = FFTdenoise_signal.drop(columns=metadata_cols).copy(deep=True)

    FFTdenoise_noise.reset_index(inplace=True)
    FFTd_N_metadata = FFTdenoise_noise[metadata_cols].copy(deep=True)
    FFTd_N_data = FFTdenoise_noise.drop(columns=metadata_cols).copy(deep=True)

    return FFTd_S_data, FFTd_S_metadata, FFTd_N_data, FFTd_N_metadata


def get_spectrum(df_data, df_metadata, sn_name, sn_phase):
    df_data_mask_i = df_metadata["SN Name"] == sn_name
    df_data_mask_i &= df_metadata["Spectrum Phase"] == sn_phase
    spectrum = df_data.loc[df_data_mask_i].values
    assert spectrum.shape[0] == 1, f"More than one spectrum called '{sn_name}' at phase '{sn_phase}'."
    return spectrum[0]


def create_SNRmetadata_file(
    SNRmetadata_savefile="../data/forSNR/SNRmetadata.parquet",
    file_metadata="../data/forSNR/metadata.parquet",
):
    if isfile(SNRmetadata_savefile):
        print(f"File '{SNRmetadata_savefile}' already exists.")
        
        decision = None
        while True:
            decision = input("Would you like to overwrite it? (y/n)").lower()
            if decision == "y":
                break
                return
            elif decision == "n":
                print("Exiting.")
                return
            else:
                print("Invalid response.")

    print(f"Creating '{SNRmetadata_savefile}'")
    df_SNRmetadata = pd.read_parquet(file_metadata)
    df_SNRmetadata["Denoising Parameter"] = np.nan
    df_SNRmetadata["minima_i"] = np.nan
    df_SNRmetadata["searchBlu"] = np.nan
    df_SNRmetadata["searchRed"] = np.nan
    df_SNRmetadata["useBlu"] = np.nan
    df_SNRmetadata["useRed"] = np.nan
    df_SNRmetadata["maxBlu"] = np.nan
    df_SNRmetadata["maxRed"] = np.nan
    df_SNRmetadata["noiseWindowBlu"] = np.nan
    df_SNRmetadata["noiseWindowRed"] = np.nan
    df_SNRmetadata.to_parquet(SNRmetadata_savefile)
    return df_SNRmetadata


def load_SNRmetadata(SNRmetadata_savefile="../data/forSNR/SNRmetadata_nodupe.parquet"):
    df_SNRmetadata = pd.read_parquet(SNRmetadata_savefile)
    return df_SNRmetadata


def write_to_SNRmetadata(
    dataset_i,
    options,
    SNRmetadata_savefile="../data/forSNR/data_metadata_SNRinfo.parquet",
):
    print(f"Writing to {SNRmetadata_savefile}...")
    df_SNRmetadata = pd.read_parquet(SNRmetadata_savefile)
    
    SNRmetadata_mask = df_SNRmetadata["name_prefix"] == dataset_i["name_prefix"]
    SNRmetadata_mask &= df_SNRmetadata["name_year"] == dataset_i["name_year"]
    SNRmetadata_mask &= df_SNRmetadata["name_suffix"] == dataset_i["name_suffix"]
    SNRmetadata_mask &= df_SNRmetadata["Spectrum Phase"] == dataset_i["Spectrum Phase"]
    SNRmetadata_mask &= df_SNRmetadata["Spectrum Cardinality"] == dataset_i["Spectrum Cardinality"]
    
    # SNRmetadata_mask &= df_SNRmetadata["Spectrum Phase"] == sn_phase
    # assert df_SNRmetadata.loc[SNRmetadata_mask].shape[0] == 1, f"More than one spectrum called '{sn_name}' at phase '{sn_phase}'."

    df_SNRmetadata.loc[SNRmetadata_mask, "Denoising Parameter"] = options["sd"]
    df_SNRmetadata.loc[SNRmetadata_mask, "minima_i"] = options["minima_i"]
    df_SNRmetadata.loc[SNRmetadata_mask, "searchBlu"] = options["searchBlu"]
    df_SNRmetadata.loc[SNRmetadata_mask, "searchRed"] = options["searchRed"]
    df_SNRmetadata.loc[SNRmetadata_mask, "useBlu"] = options["useBlu"]
    df_SNRmetadata.loc[SNRmetadata_mask, "useRed"] = options["useRed"]
    df_SNRmetadata.loc[SNRmetadata_mask, "maxBlu"] = options["maxBlu"]
    df_SNRmetadata.loc[SNRmetadata_mask, "maxRed"] = options["maxRed"]
    df_SNRmetadata.loc[SNRmetadata_mask, "noiseWindowBlu"] = options["noiseWindowBlu"]
    df_SNRmetadata.loc[SNRmetadata_mask, "noiseWindowRed"] = options["noiseWindowRed"]

    df_SNRmetadata.to_parquet(SNRmetadata_savefile)
    return


def review_spectrum(df_meta_i, wvl, spec_i, specsnr):
    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(6, 6))
    # fig.subplots_adjust(hspace=0, wspace=0.1)
    axes[0].set_xlim((4400, 7100))
    title = (
        f'{df_meta_i["SN Subtype"]} | '
        f'{df_meta_i["SN Name"]} at '
        f'{df_meta_i["Spectrum Phase"]} '
        f'({df_meta_i["Spectrum Cardinality"]})'
    )
    fig.suptitle(title, fontsize=15)

    c_spectrum = "tab:blue"
    c_signal = "tab:orange"
    c_noise = "tab:green"

    feature_location, feature_name = sf.get_spectral_feature(df_meta_i["SN Subtype"], df_meta_i["Spectrum Phase"])
    [ax.axvline(x=feature_location, c="k", ls="--") for ax in axes.ravel()]

    axes[0].set_title(rf"Gaussian Denoising | $\sigma={df_meta_i['Denoising Parameter']}$")
    
    axes[0].plot(wvl, specsnr.spectrum, c=c_spectrum)
    
    axes[1].plot(wvl, specsnr.spectrum, c=c_spectrum)
    axes[1].plot(wvl, specsnr.signal, c=c_signal)

    axes[2].plot(wvl, specsnr.signal, c=c_signal)
    
    axes[3].axhline(y=0, c="k", ls=":")
    axes[3].plot(wvl, specsnr.noise, c=c_noise)

    # sync_ylim(axes[0, 0], axes[0, 1])
    # sync_ylim(axes[1, 0], axes[1, 1])
    # sync_ylim(axes[2, 0], axes[2, 1])
    # sync_ylim(axes[3, 0], axes[3, 1])

    axes[3].set_ylim((-.25, .25))
    axes[3].yaxis.tick_right()

    axes[0].annotate(
        feature_name,
        (feature_location, axes[0].get_ylim()[1]*0.85),
        xytext=(feature_location*1.05, axes[0].get_ylim()[1]*0.85),
        va="center",
        arrowprops={
            "width": 1,
            "headwidth": 5,
            "headlength": 10,
            "color": "tab:red"
        }
    )

    axes[0].axvline(x=specsnr.line_observed, c="k", ls=":")
    axes[1].axvline(x=specsnr.line_observed, c="k", ls=":")
    axes[2].axvline(x=specsnr.line_observed, c="k", ls=":")
    axes[3].axvline(x=specsnr.line_observed, c="k", ls=":")

    if specsnr.useBlu:
        axes[3].fill_between(
            specsnr.wvl[specsnr.blu_inds],
            y1=[-1000]*specsnr.blu_inds.size,
            y2=[1000]*specsnr.blu_inds.size,
            color="tab:blue",
            alpha=0.5)

    if specsnr.useRed:
        axes[3].fill_between(
            specsnr.wvl[specsnr.red_inds],
            y1=[-1000]*specsnr.red_inds.size,
            y2=[1000]*specsnr.red_inds.size,
            color="tab:red",
            alpha=0.5)

    axes[0].plot(specsnr.pc_wvl, specsnr.pseudo_cont, c="tab:purple")
    axes[1].plot(specsnr.pc_wvl, specsnr.pseudo_cont, c="tab:purple")
    axes[2].plot(specsnr.pc_wvl, specsnr.pseudo_cont, c="tab:purple")

    fig.show()
    return fig


def sync_ylim(ax1, ax2):
    ylim_extreme = np.max(np.abs([*ax1.get_ylim(), *ax2.get_ylim()]))
    ax1.set_ylim((-ylim_extreme, ylim_extreme))
    ax2.set_ylim((-ylim_extreme, ylim_extreme))

    ax1.tick_params(axis="x", which="both", direction="in")
    ax2.tick_params(axis="x", which="both", direction="in")

    ax2.tick_params(axis="y", which="both", direction="in")
    ax2.yaxis.tick_right()
    return
    