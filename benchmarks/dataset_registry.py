"""
Grinsztajn et al. (2022) benchmark dataset registry.
"Why tree-based models still outperform deep learning on tabular data"
"""

CLASSIFICATION_DATASETS = {
    "electricity":              44120,
    "covertype":                44121,
    "pol":                      44122,
    "house_16H":                44123,
    "kdd_ipums_la_97-small":    44124,
    "MagicTelescope":           44125,
    "bank-marketing":           44126,
    "phoneme":                  44127,
    "MiniBooNE":                44128,
    "jannis":                   44129,
    "Higgs":                    44130,
    "eye_movements":            44157,
    "california":               44159,
    "heloc":                    44161,
    "credit":                   44089,
    "albert":                   44162,
    "default-of-credit-card-clients": 44090,
}

REGRESSION_DATASETS = {
    "cpu_act":              44132,
    "pol":                  44133,
    "elevators":            44134,
    "wine_quality":         44136,
    "Ailerons":             44137,
    "houses":               44138,
    "house_16H":            44139,
    "diamonds":             44140,
    "Brazilian_houses":     44141,
    "Bike_Sharing_Demand":  44142,
    "nyc-taxi-green-dec-2016": 44143,
    "fps-in-games":         44144,
    "Mercedes_Benz_Greener_Manufacturing": 44145,
    "SGEMM_GPU_kernel_performance": 44146,
    "delays_zurich_transport": 44147,
}

ALL_DATASETS = {
    "classification": CLASSIFICATION_DATASETS,
    "regression": REGRESSION_DATASETS,
}
