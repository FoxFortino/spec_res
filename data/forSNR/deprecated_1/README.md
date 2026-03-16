This directory should contain all of the data that I started with when I began manually reviewing each spectrum.

data.parquet - Only the fluxes
metadata.parquet - SN name, phase, subtype, etc.
SNRmetadata.parquet - A mostly empty file that is designed to be filled up with the parameters used to calculate the SNR for each spectrum. It's mostly empty because I created another set of files (with the suffix `_no_dupe`) that excluded any spectrum that shared the same `SN Name` and `Spectral Phase`. I wanted to keep track of which ones were removed so I kept this data here.

All of these files *should* share a common index.