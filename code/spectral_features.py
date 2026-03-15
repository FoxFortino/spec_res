import numpy as np

TYPES_Ia = ["Ia-norm", "Ia-91T", "Ia-91bg", "Iax"]
TYPES_Ib = ["Ib-norm", "Ibn", "IIb"]
TYPES_Ic = ["Ic-norm", "Ic-broad"]
TYPES_II = ["IIP"]


def get_spectral_feature(sn_subtype, sn_phase):
    sn_maintype = get_maintype_from_subtype(sn_subtype)

    if sn_subtype == "Iax":
        if sn_phase <= 7:
            return 5129, "FeIII"
        elif sn_phase > 7:
            return 5169, "FeII"

    elif sn_subtype == "IIb":
        if sn_phase <= 0:
            return 6562, r"H$\alpha$"
        elif sn_phase > 0:
            return 5876, "HeI"

    elif sn_subtype == "Ic-norm":
        if -np.inf <= sn_phase <= np.inf:
            return 5895, "NaI D"

    elif sn_subtype == "Ic-broad":
        if -np.inf <= sn_phase <= np.inf:
            return 5895, "NaI D"

    if sn_maintype == "Ia":
        if -np.inf <= sn_phase <= 20:
            return 6355, "SiII"
        if sn_phase > 20:
            return 6355-500, "Triplet"

    if sn_maintype == "Ib":
        if -np.inf <= sn_phase <= np.inf:
            return 5876, "HeI"

    if sn_maintype == "Ic":
        if -np.inf <= sn_phase <= np.inf:
            return 5169, "FeII"

    if sn_maintype == "II":
        if -np.inf <= sn_phase <= np.inf:
            return 6562, r"H$\alpha$"


def get_maintype_from_subtype(subtype):
    if subtype in TYPES_Ia:
        maintype = "Ia"
    elif subtype in TYPES_Ib:
        maintype = "Ib"
    elif subtype in TYPES_Ic:
        maintype = "Ic"
    elif subtype in TYPES_II:
        maintype = "II"
    else:
        assert False, f"SN subtype '{subtype}' not known."

    return maintype

