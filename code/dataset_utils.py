import sys

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


def load_dataset(
    file_dataset="../data/forSNR/dataset.parquet",
    file_signal_only="../data/forSNR/signal.parquet",
    file_noise_only="../data/forSNR/noise.parquet",
):
    df_dataset = pd.read_parquet(file_dataset)
    df_signal = pd.read_parquet(file_signal_only)
    df_noise = pd.read_parquet(file_noise_only)
    
    wvl, spectra, df_meta = unpack_dataset(df_dataset)
    wvl_s, spectra_signal, df_meta_signal = unpack_dataset(df_signal)
    wvl_n, spectra_noise, df_meta_noise = unpack_dataset(df_noise)
    
    assert np.all(wvl == wvl_s)
    assert np.all(wvl == wvl_n)
    del wvl_s, wvl_n
    
    assert_frame_equal(df_meta, df_meta_signal)
    assert_frame_equal(df_meta, df_meta_noise)
    del df_meta_signal, df_meta_noise
    
    return wvl, df_meta, spectra, spectra_signal, spectra_noise


def unpack_dataset(sn_data):
    df_fluxes = sn_data.filter(regex="\d+")
    fluxes = df_fluxes.to_numpy(dtype=float)

    flux_columns = df_fluxes.columns
    wvl = flux_columns.to_numpy(dtype=float)

    df_metadata = sn_data.drop(columns=flux_columns)

    return wvl, fluxes, df_metadata
