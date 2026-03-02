import sys
from os.path import join, isfile

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import display, clear_output

import spectral_features as sf
import measure_signal as ms

from icecream import ic


def review(i=0):
    df_data, df_metadata, wvl = load_sn_data()
    FFTd_S_data, FFTd_S_metadata, FFTd_N_data, FFTd_N_metadata = load_FFTdenoised_data()

    sort_cols = ["SN Subtype", "SN Name", "Spectral Phase"]
    df_metadata_sorted = df_metadata.sort_values(sort_cols).copy(deep=True)

    options = reset_options()

    while True:            
        sn_name = df_metadata_sorted["SN Name"].iloc[i]
        sn_type = df_metadata_sorted["SN Subtype"].iloc[i]
        sn_phase = df_metadata_sorted["Spectral Phase"].iloc[i]
        print(f"{sn_type} | {sn_name} | {sn_phase} | index: {i}")

        df_SNRmetadata = load_SNRmetadata()
        SNRmetadata_mask = df_SNRmetadata["SN Name"] == sn_name
        SNRmetadata_mask &= df_SNRmetadata["Spectral Phase"] == sn_phase
        assert df_SNRmetadata.loc[SNRmetadata_mask].shape[0] == 1, f"More than one spectrum called '{sn_name}' at phase '{sn_phase}'."
        loaded = False
        if not np.isnan(df_SNRmetadata.loc[SNRmetadata_mask, "Denoising Parameter"].values[0]):
            print(f"Loading saved parameters...")
            if np.isnan(df_SNRmetadata.loc[SNRmetadata_mask, "minima_i"].values[0]):
                options["minima_i"] = None
            else:
                options["minima_i"] = df_SNRmetadata.loc[SNRmetadata_mask, "minima_i"].values[0]

            options["sd"] = df_SNRmetadata.loc[SNRmetadata_mask, "Denoising Parameter"].values[0]
            options["searchBlu"] = df_SNRmetadata.loc[SNRmetadata_mask, "searchBlu"].values[0]
            options["searchRed"] = df_SNRmetadata.loc[SNRmetadata_mask, "searchRed"].values[0]
            options["useBlu"] = df_SNRmetadata.loc[SNRmetadata_mask, "useBlu"].values[0]
            options["useRed"] = df_SNRmetadata.loc[SNRmetadata_mask, "useRed"].values[0]
            options["maxBlu"] = int(df_SNRmetadata.loc[SNRmetadata_mask, "maxBlu"].values[0])
            options["maxRed"] = int(df_SNRmetadata.loc[SNRmetadata_mask, "maxRed"].values[0])
            options["noiseWindowBlu"] = df_SNRmetadata.loc[SNRmetadata_mask, "noiseWindowBlu"].values[0]
            options["noiseWindowRed"] = df_SNRmetadata.loc[SNRmetadata_mask, "noiseWindowRed"].values[0]
            loaded = True

        # Try to extract the FFT-denoised version of the spectrum corresponding
        # to `sn_name` and `sn_phase`. If it doesn't exist, set the signal and
        # noise to 0 and carry on.
        try:
            FFTd_signal = get_spectrum(FFTd_S_data, FFTd_S_metadata, sn_name, sn_phase)
            FFTd_noise = get_spectrum(FFTd_N_data, FFTd_N_metadata, sn_name, sn_phase)
        except IndexError:
            print(f"Supernova '{sn_name}' at phase {sn_phase} does not have a FFT-denoised sample.")
            FFTd_signal = [0] * wvl.size
            FFTd_noise = [0] * wvl.size

        # Initialize the spectrumSNR object with the desired spectrum info.
        spectrum = get_spectrum(df_data, df_metadata, sn_name, sn_phase)
        specsnr = ms.SpectrumSNR(
            sn_name,
            sn_type,
            sn_phase,
            wvl,
            spectrum)
        specsnr.minmax_normalize()
        specsnr.set_spectral_feature()  # See spectral_features.py

        # If the `sd` is set to -1 that means we have selected the FFT-denoised
        # version. Instead of using `specsnr.denoise_gaussian`, manuallyy set
        # the denoising parameter, signal and noise.
        if options["sd"] == -1:
            specsnr.denoising_parameter = -1
            specsnr.signal = FFTd_signal
            specsnr.noise = FFTd_noise
        else:
            specsnr.denoise_gaussian(stddev=options["sd"])

        # Proceed with the SNR calculation algorithm.
        specsnr.find_spectral_line(
            feature_search_bounds=(options["searchBlu"], options["searchRed"]),
            minima_i=options["minima_i"],
            plot=True)
        display(plt.gcf())
        specsnr.find_spectral_shoulders(
            blu_shoulder_nudge=options["maxBlu"],
            red_shoulder_nudge=options["maxRed"],
        )
        specsnr.calc_pEW()
        specsnr.measure_feature_noise(
            noise_window_blu=options["noiseWindowBlu"],
            noise_window_red=options["noiseWindowRed"],
            useBlu=options["useBlu"],
            useRed=options["useRed"],
        )
        specsnr.measure_SNR(plot=True)
        display(plt.gcf())

        specsnr.spectrum *= (specsnr.spec_max - specsnr.spec_min)
        specsnr.spectrum += specsnr.spec_min

        specsnr.signal *= (specsnr.spec_max - specsnr.spec_min)
        specsnr.signal += specsnr.spec_min

        review_spectrum(
            sn_name,
            sn_phase,
            sn_type,
            wvl,
            spectrum, 
            FFTd_signal,
            FFTd_noise,
            specsnr.signal,
            specsnr.noise,
            options["sd"],
        )
        display(plt.gcf())

        print(f"{sn_type} | {sn_name} | {sn_phase} | index: {i}")

        if loaded:
            print(f"Loaded parameters from file.")
        user_input = input()
        clear_output(wait=True)
        plt.close()
        plt.close()
        plt.close()

        # If the user simply hits enter then we increment i and save the
        # options to the SNRmetadata file.
        if user_input == "":
            write_to_SNRmetadata(sn_name, sn_phase, options)
            options = reset_options()
            i += 1
        else:
            i, options = logic(i, user_input, options)


        

def logic(i, user_input, options):
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

        return i, options

    else:
        try:
            sd = float(user_input)
            options["sd"] = sd
            return i, options
        except ValueError:
            print(f"Invalid command.")
            return i, options


def reset_options():
    options = {
        "sd": 10,
        "minima_i": None,
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


def load_sn_data(
    file_data="../data/forSNR/data.parquet",
    file_metadata="../data/forSNR/metadata.parquet",
):
    df_data = pd.read_parquet(file_data)
    df_metadata = pd.read_parquet(file_metadata)
    wavelengths = df_data.columns.values.astype(float)
    return df_data, df_metadata, wavelengths


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
        "Spectral Phase",
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
    df_data_mask_i &= df_metadata["Spectral Phase"] == sn_phase
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


def load_SNRmetadata(SNRmetadata_savefile="../data/forSNR/SNRmetadata.parquet"):
    df_SNRmetadata = pd.read_parquet(SNRmetadata_savefile)
    return df_SNRmetadata


def write_to_SNRmetadata(
    sn_name,
    sn_phase,
    options,
    SNRmetadata_savefile="../data/forSNR/SNRmetadata.parquet",
):
    print(f"Writing to {SNRmetadata_savefile}...")
    df_SNRmetadata = pd.read_parquet(SNRmetadata_savefile)
    SNRmetadata_mask = df_SNRmetadata["SN Name"] == sn_name
    SNRmetadata_mask &= df_SNRmetadata["Spectral Phase"] == sn_phase
    assert df_SNRmetadata.loc[SNRmetadata_mask].shape[0] == 1, f"More than one spectrum called '{sn_name}' at phase '{sn_phase}'."

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


def review_spectrum(
    sn_name,
    sn_phase,
    sn_type,
    wvl, 
    spectrum, 
    FFTd_signal,
    FFTd_noise,
    gaussian_signal,
    gaussian_noise,
    sd,
):

    fig, axes = plt.subplots(nrows=4, ncols=2, sharex=True, figsize=(12, 7))
    fig.subplots_adjust(hspace=0, wspace=0.1)
    axes[0, 0].set_xlim((4400, 7100))

    fig.suptitle(f"{sn_type} | {sn_name} at {sn_phase}", fontsize=15)

    c_spectrum = "tab:blue"
    c_signal = "tab:orange"
    c_noise = "tab:green"

    feature_location, feature_name = sf.get_spectral_feature(sn_type, sn_phase)
    [ax.axvline(x=feature_location, c="k", ls="--") for ax in axes.ravel()]

    axes[0, 0].set_title("FFT Denoising")
    axes[0, 1].set_title(rf"Gaussian Denoising | $\sigma={sd}$")
    
    axes[0, 0].plot(wvl, spectrum, c=c_spectrum)
    axes[0, 1].plot(wvl, spectrum, c=c_spectrum)
    
    axes[1, 0].plot(wvl, spectrum, c=c_spectrum)
    axes[1, 1].plot(wvl, spectrum, c=c_spectrum)
    
    axes[1, 0].plot(wvl, FFTd_signal, c=c_signal)
    axes[1, 1].plot(wvl, gaussian_signal, c=c_signal)

    axes[2, 0].plot(wvl, FFTd_signal, c=c_signal)
    axes[2, 1].plot(wvl, gaussian_signal, c=c_signal)
    
    axes[3, 0].axhline(y=0, c="k", ls=":")
    axes[3, 1].axhline(y=0, c="k", ls=":")
    axes[3, 0].plot(wvl, FFTd_noise, c=c_noise)
    axes[3, 1].plot(wvl, gaussian_noise, c=c_noise)
    # axes[3, 0].axvline(x=wvl_shoulder_blu, c="tab:blue")
    # axes[3, 1].axvline(x=wvl_shoulder_red, c="tab:red")

    sync_ylim(axes[0, 0], axes[0, 1])
    sync_ylim(axes[1, 0], axes[1, 1])
    sync_ylim(axes[2, 0], axes[2, 1])
    # sync_ylim(axes[3, 0], axes[3, 1])

    axes[3, 0].set_ylim((-.5, .5))
    axes[3, 1].set_ylim((-.5, .5))
    axes[3, 1].yaxis.tick_right()

    axes[0, 0].annotate(
        feature_name,
        (feature_location, axes[0, 0].get_ylim()[1]*0.85),
        xytext=(feature_location*1.05, axes[0, 0].get_ylim()[1]*0.85),
        va="center",
        arrowprops={
            "width": 1,
            "headwidth": 5,
            "headlength": 10,
            "color": "tab:red"
        }
    )

    axes[0, 1].annotate(
        feature_name,
        (feature_location, axes[0, 1].get_ylim()[1]*0.85),
        xytext=(feature_location*1.05, axes[0, 1].get_ylim()[1]*0.85),
        va="center",
        arrowprops={
            "width": 1,
            "headwidth": 5,
            "headlength": 10,
            "color": "tab:red"
        }
    )

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
    