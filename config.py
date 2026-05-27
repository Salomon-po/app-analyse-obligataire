import os
# ─── PAYS ─────────────────────────────────────────────────────────────────────
PAYS_SUPPORTES = []

# ─── TYPES D'OBLIGATIONS ──────────────────────────────────────────────────────
TYPES_OBLIGATIONS = [
    'US Treasury', 'T-bond', 'CORP',
    'Bund', 'Bobl', 'Schatz', 'Bubill', 'Green issue',
    'Treasury Gilt', 'Treasury Stock', 'Green Gilt',
    'Index-linked Treasury Gilt', 'Index-Linked Treasury Gilt',
    'JGB', 'adju_LT', 'adju_MT', 'adju_I'
]
# ─── FRÉQUENCES ───────────────────────────────────────────────────────────────
FREQ_MAP = {
    'US Treasury': 2, 'T-bond': 2, 'CORP': 2,
    'Treasury Gilt': 2, 'Treasury Stock': 2, 'Green Gilt': 2,
    'Index-linked Treasury Gilt': 2, 'Index-Linked Treasury Gilt': 2,
    'JGB': 2,
    'Bund': 1, 'Green issue': 1, 'Bubill': 1,
    'Bobl': 1, 'Schatz': 1,
    'adju_LT': 1, 'adju_MT': 1, 'adju_I': 1,
}

FREQ_PAR_PAYS = {
    'USA': 2, 'DE': 1, 'France': 1,
    'UK': 2, 'Japon': 2, 'Maroc': 1,
}
# ─── DEVISES ──────────────────────────────────────────────────────────────────
DEVISES = ['USD', 'EUR', 'GBP', 'JPY', 'MAD']

# ─── SPREADS ──────────────────────────────────────────────────────────────────
SPREADS = {
    'USA': 0.000, 'DE': -0.015, 'France': -0.010,
    'UK': -0.005, 'Japon': -0.030, 'Maroc': 0.005,
}
# ─── BANQUES CENTRALES ────────────────────────────────────────────────────────
BANQUES_CENTRALES = {
    'USA': 'FED', 'DE': 'BCE', 'France': 'BCE',
    'UK': 'BOE', 'Japon': 'BOJ', 'Maroc': 'BAM',
}

ANTICIPATIONS_BC_DEFAUT = {
    'USA': -0.0025, 'DE': -0.0050, 'France': -0.0050,
    'UK': -0.0025, 'Japon': +0.0025, 'Maroc': -0.0025,
}
# ─── SCÉNARIOS ────────────────────────────────────────────────────────────────
SCENARIOS = {
    'Optimiste'    : {'USA': -0.01,  'DE': -0.015, 'France': -0.015,
                      'UK': -0.01,  'Japon': -0.005, 'Maroc': -0.01},
    'Modéré'       : {'USA': +0.005, 'DE': -0.0025, 'France': -0.0025,
                      'UK': +0.0025, 'Japon': +0.005, 'Maroc': +0.0025},
    'Adverse'      : {'USA': +0.01,  'DE': +0.005, 'France': +0.005,
                      'UK': +0.01,  'Japon': +0.01, 'Maroc': +0.01},
    'Très adverse' : {'USA': +0.02,  'DE': +0.015, 'France': +0.015,
                      'UK': +0.02,  'Japon': +0.02, 'Maroc': +0.02},
}
# ─── COULEURS ─────────────────────────────────────────────────────────────────
COLORS = {
    'primary'   : '#1F4E79',
    'secondary' : '#2E74B5',
    'light'     : '#EBF3FB',
    'positive'  : '#375623',
    'negative'  : '#C00000',
    'warning'   : '#C55A11',
    'neutral'   : '#808080',
}
COLORS_PAYS = {
    'USA': '#1F4E79', 'DE': '#F0C040',
    'France': '#2E74B5', 'UK': '#C00000',
    'Japon': '#E07040', 'Maroc': '#375623',
}

COLORS_RISK = {
    'Low': '#375623', 'Medium': '#C55A11', 'High': '#C00000',
}
# ─── ARIMA ────────────────────────────────────────────────────────────────────
ARIMA_ORDER  = (0, 1, 1)
ARIMA_START  = '2015-01-01'
ARIMA_END    = '2025-12-31'
ARIMA_TICKER = '^TNX'
# ─── ML ───────────────────────────────────────────────────────────────────────
N_CLUSTERS        = 3
LINKAGE           = 'ward'
METRIC_CLUST      = 'euclidean'
CONTAMINATION_ISO = 0.05
FEATURES_ML       = ['ytm_calculee', 'duration_mod', 'convexity', 'var_prix_sens_convex']

# ─── OPTIMISATION ─────────────────────────────────────────────────────────────
POIDS_MAX    = 0.05
DURATION_MIN = 0.5
# ─── COLONNES ─────────────────────────────────────────────────────────────────
COLS_OBLIGATOIRES = ['isin', 'Coupon', 'date_maturite', 'prix', 'Pays']
COLS_OPTIONNELLES = {
    'type_obligation': 'US Treasury',
    'date_emission'  : '2020-01-01',
    'ytm'            : None,
    'benchmark'      : None,
    'devise'         : 'USD',
}
# ─── ADMIN ────────────────────────────────────────────────────────────────────
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")