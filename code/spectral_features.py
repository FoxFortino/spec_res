import numpy as np


def get_spectral_feature(sn_subtype, sn_phase):
    sn_maintype = sn_subtype[:2]

    if sn_subtype == "Ia-norm":
        if sn_phase > 20:
            return 6355-500, "Triplet"
    
    if sn_maintype == "Ia":
        if -np.inf <= sn_phase <= np.inf:
            return 6355, "SiII"

    if sn_maintype == "Ib":
        if -np.inf <= sn_phase <= np.inf:
            return 5876, "HeI"

    if sn_maintype == "Ic":
        if -np.inf <= sn_phase <= np.inf:
            return 5169, "FeII"

    if sn_maintype == "II":
        if -np.inf <= sn_phase <= np.inf:
            return 6562, r"H$\alpha$"