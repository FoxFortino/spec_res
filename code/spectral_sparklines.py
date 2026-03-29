import sys
import os
from os.path import isdir
from os.path import join
from shutil import rmtree
from glob import glob
import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from scipy import stats
from matplotlib import pyplot as plt

import dataset_utils as du
import measure_signal as ms

RNG = np.random.default_rng(1415)
DIR_DATA = "../data/forSNR/"
FILE_DATASET = join(DIR_DATA, "dataset.parquet")
FILE_SIGNAL = join(DIR_DATA, "signal.parquet")
FILE_NOISE = join(DIR_DATA, "noise.parquet")


def main(dir_figs, num_sparklines_per_fig, new_SNR):
    dataset = load_datasets()
    wvl = dataset["wvl"]
    df_meta = dataset["df_meta"]
    spectra = dataset["spectra"]
    num_spectra = dataset["num_spectra"]
    num_wvl = dataset["num_wvl"]

    subtypes_ID = df_meta["SN Subtype ID"].unique()
    subtypes_ID_to_str = df_meta.groupby(by="SN Subtype ID")["SN Subtype"]
    subtypes_ID_to_str = subtypes_ID_to_str.apply(lambda x: x.to_numpy()[0])
    subtypes_ID_to_str = dict(subtypes_ID_to_str)
    subtypes_ID_to_dir = make_dirs(dir_figs, subtypes_ID_to_str)

    new_noise = None
    if new_SNR is not None:
        new_noise = dataset_make_new_noise(
            new_SNR,
            df_meta,
            num_spectra,
            num_wvl,
            RNG)

    dataset_calc_SNR(
        new_SNR,
        num_spectra,
        wvl,
        df_meta,
        spectra,
        new_noise)

    df_meta = dataset_sort(df_meta)

    for subtype_ID, subtype_str in subtypes_ID_to_str.items():
        dir_subtype_figs = subtypes_ID_to_dir[subtype_ID]
        
        mask = (df_meta["SN Subtype ID"] == subtype_ID)
        df_subtype = df_meta[mask].copy(deep=True).reset_index(drop=True)
        
        subtype_num_spectra = mask.sum()
        assert subtype_num_spectra == df_subtype.shape[0]
        print(subtype_ID, subtype_str, subtype_num_spectra)
        for i in tqdm(range(0, subtype_num_spectra, num_sparklines_per_fig)):
            num_plots = np.min((subtype_num_spectra - i, num_sparklines_per_fig))
            
            fig, axes = plt.subplots(
                ncols=1,
                nrows=num_plots,
                figsize=(8, 1.5*num_plots))
    
            for j in range(num_plots):
                if num_plots == 1:
                    ax = axes
                else:
                    ax = axes[j]

                plot_sparkline(ax, df_subtype.loc[i+j, "specsnr"])
            
            fig.tight_layout()
            fig.savefig(join(dir_subtype_figs, f"{i:0>5}"))
            plt.close()


def load_datasets():
    df_dataset = pd.read_parquet(FILE_DATASET)
    df_signal = pd.read_parquet(FILE_SIGNAL)
    df_noise = pd.read_parquet(FILE_NOISE)

    wvl, spectra_dataset, df_meta = du.unpack_dataset(df_dataset)
    _1_wvl, spectra_signal, df_meta_signal = du.unpack_dataset(df_signal)
    _2_wvl, spectra_noise, df_meta_noise = du.unpack_dataset(df_noise)
    assert np.all(wvl == _1_wvl)
    assert np.all(wvl == _2_wvl)
    del _1_wvl, _2_wvl
    assert_frame_equal(df_meta, df_meta_signal)
    assert_frame_equal(df_meta, df_meta_noise)
    del df_meta_signal, df_meta_noise
    
    assert wvl.ndim == 1, (wvl.ndim, wvl.shape)
    num_wvl = wvl.size
    num_spectra = spectra_dataset.shape[0]

    dataset = {
        "df_dataset": df_dataset,
        "df_signal": df_signal,
        "df_noise": df_noise,
        "wvl": wvl,
        "df_meta": df_meta,
        "spectra": spectra_dataset,
        "signal": spectra_signal,
        "noise": spectra_noise,
        "num_wvl": num_wvl,
        "num_spectra": num_spectra,
    }
    return dataset


def dataset_make_new_noise(new_SNR, df_meta, num_spectra, num_wvl, rng):
    new_N = (df_meta["S (SNR)"] / new_SNR).to_numpy()
    
    new_N_arr = np.full(
        (num_spectra, num_wvl),
        new_N[..., np.newaxis],
    )
    
    new_noise = stats.norm.rvs(loc=0, scale=new_N_arr, random_state=rng)
    return new_noise


def dataset_calc_SNR(new_SNR, num_spectra, wvl, df_meta, spectra, new_noise):
    print(f"Calculating SNR...")
    for i in tqdm(range(num_spectra)):
        specsnr = ms.SpectrumSNR(
            df_meta.loc[i, "SN Name"],
            df_meta.loc[i, "SN Subtype"],
            df_meta.loc[i, "Spectrum Phase"],
            wvl,
            spectra[i])
        
        if new_noise is not None:
            specsnr.execute_algorithm(df_meta.loc[i], new_noise=new_noise[i])
        else:
            specsnr.execute_algorithm(df_meta.loc[i])
            
        df_meta.loc[i, "specsnr"] = specsnr
    return


def dataset_sort(df_meta):
    df_meta["post_injection_range"] = [(specsnr.signal + specsnr.noise).max() - (specsnr.signal + specsnr.noise).min() for specsnr in df_meta["specsnr"]]

    df_meta = df_meta.sort_values(
        by=["SN Subtype ID", "post_injection_range"], ascending=[True, False]).reset_index(drop=True)

    return df_meta


def make_dirs(dir_figs, subtypes_ID_to_str):
    print(f"Saving figures at: '{dir_figs}'")
    
    if isdir(dir_figs):
        total_files = sum(len(files) for _, _, files in os.walk(dir_figs))

        if total_files != 0:
            ans = input(f"Confirm overwriting {total_files} file(s)? (y/N) ")
            if ans == "" or ans.lower() == "n":
                sys.exit(0)
            elif ans.lower() == "y":
                rmtree(dir_figs)
                print(f"Removed {total_files} files.")
        else:
            rmtree(dir_figs)
            

    os.mkdir(dir_figs)

    subtypes_ID_to_dir = {}
    
    for subtype_ID, subtype_str in subtypes_ID_to_str.items():
        dir_subtype_figs = join(dir_figs, f"{subtype_ID:0>2}_{subtype_str}")
        os.mkdir(dir_subtype_figs)
        subtypes_ID_to_dir[subtype_ID] = dir_subtype_figs

    return subtypes_ID_to_dir


def plot_sparkline(ax, specsnr):
    ax.plot(specsnr.wvl, specsnr.signal + specsnr.noise, c="k", lw=1)
    ax.plot(specsnr.wvl, specsnr.signal, c="tab:blue")
    ax.plot(specsnr.pc_wvl, specsnr.pseudo_cont, c="tab:green")
    
    ax.set_xlim((4500, 7000))
    # ax.axis("off")
    ax.axhline(y=0, c="k", ls=":")
    ax.axhline(y=1, c="k", ls=":")
    ax.get_xaxis().set_visible(False)
    
    y_mid = np.sum(ax.get_ylim()) / 2
    x_lo, x_hi = ax.get_xlim()

    text_id = (
        f"{specsnr.name}\n"
        f"{specsnr.subtype}\n"
        f"{specsnr.phase}"
    )

    spec_max = (specsnr.signal + specsnr.noise).max()
    spec_min = (specsnr.signal + specsnr.noise).min()
    text_info = (
        f"$SNR = {specsnr.SNR:.2f}$" "\n"
        f"$S = {specsnr.S:.2e}$" "\n"
        f"$N = {specsnr.N:.2e}$" "\n"
        f"$\sigma = {specsnr.denoising_parameter}$" "\n"
        # f"Max = {spec_max:.2f}" "\n"
        # f"Min = {spec_min:.2f}" "\n"
        f"Range = {spec_max-spec_min:.2f}" #"\n"
        
    )

    ax.text(x_lo*0.90, y_mid, text_id, ha="right", va="center")
    ax.text(x_hi*1.025, y_mid, text_info, ha="left", va="center")

    if specsnr.useBlu:
        ax.axvspan(
            specsnr.wvl[specsnr.blu_inds][-1],
            specsnr.wvl[specsnr.blu_inds][0],
            color="tab:blue",
            alpha=0.25)

    if specsnr.useRed:
        ax.axvspan(
            specsnr.wvl[specsnr.red_inds][0],
            specsnr.wvl[specsnr.red_inds][-1],
            color="tab:red",
            alpha=0.25)

    return ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_figs")
    parser.add_argument("--num_sparklines_per_fig", type=int, default=10)
    parser.add_argument("--new_SNR", type=float, default=None)
    args = parser.parse_args()
    
    main(args.dir_figs, args.num_sparklines_per_fig, args.new_SNR)
    sys.exit(0)