This directory contains the data after I had manually reviewed every spectrum.

data_nodupe.parquet - The fluxes for each spectrum
metadata_nodupe.parquet - The metadata for each spectrum (SN Name, phase, subtype, etc.)
SNRmetadata_nodupe.parquet - The parameters for calculating the SNR for each spectrum.

data_metadata_SNRinfo.parquet - The combination of the other three files into one.

Each of these files *should* share a common index.