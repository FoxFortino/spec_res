def unpack_dataset(sn_data):
    df_fluxes = sn_data.filter(regex="\d+")
    fluxes = df_fluxes.to_numpy(dtype=float)

    flux_columns = df_fluxes.columns
    wvl = flux_columns.to_numpy(dtype=float)

    metadata_columns = sn_data.columns.difference(flux_columns)
    df_metadata = sn_data[metadata_columns]

    return wvl, fluxes, df_metadata
