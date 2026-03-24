The following three files contain supernova spectra that have been manually reviewed, one by one, by me, Willow Fox Fortino. I have ensured that any defective spectra (e.g., spectra with mising sections) have been removed. I have also standardized the naming convention for each spectrum.

dataset.parquet - Contains fluxes, metadata, SNR calculation parameters and the calculated S, N and SNR for each spectrum.
signal.parquet - Contains the metadata, SNR calculation parameters, the calculated S, N and SNR, and the signal of each spectrum.
noise.parquet - Contains the metadata, SNR calculation parameters, the calculated S, N and SNR, and the noise of each spectrum.

There were previously some spectra that had the same `SN Name` and `Spectral Phase` as another spectrum. This implied that they were duplicates. I manually reviewed each one and found that all but two pairs didn't actually contain duplicate data. I have denoted these in the column `Spectrum Cardinality`. Most spectra have a `Spectrum Cardinality` value of 1. Some have a value of 2 indicating that there is or was another spectrum of the same name and phase.

I also discovered that some supernovae may have different names but represent the same supernova, for example 'sn94D' and 'sn1994D'. Both of these would refer to the same supernova but have otherwise been counted has different ones. I have rectified this by standardizing the naming scheme. The original name from the `.lnw` file where I originally retrieved these spectra can be found under the column `SN Name`. The new naming system is identified by the three columns `name_prefix`, `name_year` and `name_suffix`. (CORRECTION: I have now replaced the `SN Name` column with the combination of those three `name_` columns.)

The best way to organize the dataset, in my opinion, is with the pandas command:

```
df.sort_values([
    "SN Subtype ID",
    "SN Name",
    "Spectral Phase",
    "Spectrum Cardinality"]).reset_index(drop=True)
```
