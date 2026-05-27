import numpy as np
import pandas as pd
from scipy.optimize import brentq
from config import FREQ_MAP, FREQ_PAR_PAYS

# ─── YTM ──────────────────────────────────────────────────────────────────────
def calculate_ytm(price, coupon_rate, maturity_years, face_value=100, freq=1):
    if price <= 0:
        return np.nan
    if coupon_rate == 0:
        if maturity_years == 0:
            return np.nan
        return (face_value / price) ** (1 / abs(maturity_years)) - 1
    n = int(round(abs(maturity_years) * freq))
    if n <= 0:
        return np.nan
    coupon = (coupon_rate * face_value) / freq
    def price_diff(r):
        if r <= -1:
            return 1e10
        periods = np.arange(1, n + 1)
        pv_coupons = np.sum(coupon / (1 + r) ** periods)
        pv_face    = face_value / (1 + r) ** n
        return pv_coupons + pv_face - price
    try:
        return brentq(price_diff, -0.999, 10.0)
    except ValueError:
        return np.nan

# ─── DURATION ─────────────────────────────────────────────────────────────────
def calculate_duration(price, coupon_rate, ytm_calculee,
                       maturity_years, face_value=100, freq=1):
    if price <= 0:
        return np.nan, np.nan
    if coupon_rate == 0:
        duration_mac = abs(maturity_years)
        duration_mod = duration_mac / (1 + ytm_calculee)
        return duration_mac, duration_mod
    n = int(round(abs(maturity_years) * freq))
    if n <= 0:
        return np.nan, np.nan
    coupon   = (coupon_rate * face_value) / freq
    periodes = np.arange(1, n + 1)
    pv_flux  = coupon / (1 + ytm_calculee) ** periodes
    pv_flux[-1] += face_value / (1 + ytm_calculee) ** n
    duration_mac = np.sum((periodes / freq) * pv_flux) / price
    duration_mod = duration_mac / (1 + ytm_calculee)
    return duration_mac, duration_mod

# ─── CONVEXITÉ ────────────────────────────────────────────────────────────────
def calculate_convexite(price, coupon_rate, ytm_calculee,
                        maturity_years, face_value=100, freq=1):
    if price <= 0:
        return np.nan
    n = int(round(abs(maturity_years) * freq))
    if n <= 0:
        return np.nan
    if coupon_rate == 0:
        return (n / freq) * (n / freq + 1) * face_value / (
            (1 + ytm_calculee) ** (n + 2) * price)
    coupon   = (coupon_rate * face_value) / freq
    periodes = np.arange(1, n + 1)
    pv_flux  = coupon / (1 + ytm_calculee) ** (periodes + 2)
    return np.sum((periodes / freq) * (periodes / freq + 1) * pv_flux) / price

# ─── PREPROCESSING COMPLET ────────────────────────────────────────────────────
def preprocess_dataset(df):
    df = df.copy()

    # ── Dates ─────────────────────────────────────────────────────────────────
    df['date_maturite'] = pd.to_datetime(df['date_maturite'])
    if 'date_emission' in df.columns:
        df['date_emission'] = pd.to_datetime(df['date_emission'])

    # ── Coupon → décimal ──────────────────────────────────────────────────────
    if df['Coupon'].max() > 1:
        df['Coupon'] = df['Coupon'] / 100

    # ── Maturité résiduelle ───────────────────────────────────────────────────
    today = pd.Timestamp.today()
    df['maturity_years'] = (df['date_maturite'] - today).dt.days / 365.25

    # ── Statut ────────────────────────────────────────────────────────────────
    df['statut'] = np.where(df['maturity_years'] > 0, 'active', 'expire')

    # ── Fréquence ─────────────────────────────────────────────────────────────
    if 'type_obligation' in df.columns:
        df['type_obligation'] = df['type_obligation'].str.strip()
        df['freq'] = df['type_obligation'].map(FREQ_MAP)
    if 'freq' not in df.columns:
        df['freq'] = np.nan
    df['freq'] = df['freq'].fillna(
       df['Pays'].map(FREQ_PAR_PAYS)
    ).fillna(1).astype(int) 
    
    # Avertissement pour pays inconnus
    pays_inconnus = df[~df['Pays'].isin(FREQ_PAR_PAYS.keys())]['Pays'].unique()
    if len(pays_inconnus) > 0:
        print(f"Pays non mappés (fréquence annuelle par défaut) : {pays_inconnus}")
        
    # ── YTM recalculé ─────────────────────────────────────────────────────────
    df['ytm_calculee'] = np.nan

    df.loc[df['statut'] == 'active', 'ytm_calculee'] = \
        df[df['statut'] == 'active'].apply(
            lambda row: calculate_ytm(
                price          = row['prix'],
                coupon_rate    = row['Coupon'],
                maturity_years = row['maturity_years'],
                freq           = row['freq']
            ), axis=1
        )

    if 'ytm' in df.columns:
        df.loc[df['statut'] == 'expire', 'ytm_calculee'] = \
            df.loc[df['statut'] == 'expire', 'ytm']

    df['ytm_calculee'] = df['ytm_calculee'].fillna(
        df.groupby('Pays')['ytm_calculee'].transform('median')
    )

    # ── Duration ──────────────────────────────────────────────────────────────
    def get_duration(row):
        mat = row['maturity_years'] if row['statut'] == 'active' \
              else abs(row['maturity_years'])
        return pd.Series(calculate_duration(
            price          = row['prix'],
            coupon_rate    = row['Coupon'],
            ytm_calculee   = row['ytm_calculee'],
            maturity_years = mat,
            freq           = row['freq']
        ))

    dur_results = df.apply(get_duration, axis=1)
    df['duration_mac'] = dur_results[0]
    df['duration_mod'] = dur_results[1]

    df['duration_mac'] = df.groupby('Pays')['duration_mac'].transform(
        lambda x: x.fillna(x.median()))
    df['duration_mod'] = df.groupby('Pays')['duration_mod'].transform(
        lambda x: x.fillna(x.median()))

    # ── Convexité ─────────────────────────────────────────────────────────────
    def get_convexite(row):
        mat = row['maturity_years'] if row['statut'] == 'active' \
              else abs(row['maturity_years'])
        return calculate_convexite(
            price          = row['prix'],
            coupon_rate    = row['Coupon'],
            ytm_calculee   = row['ytm_calculee'],
            maturity_years = mat,
            freq           = row['freq']
        )

    df['convexity'] = df.apply(get_convexite, axis=1)
    df['convexity'] = df.groupby('Pays')['convexity'].transform(
        lambda x: x.fillna(x.median()))

    # ── Sensibilité et variation prix ─────────────────────────────────────────
    shock = 0.01
    df['sensibility']          = -df['duration_mod']
    df['variation_prix']       = df['sensibility'] * shock
    df['var_prix_sens_convex'] = (df['sensibility'] * shock +
                                   0.5 * df['convexity'] * shock**2)

    # ── Risk level manuel ─────────────────────────────────────────────────────
    def risk_level(d):
        if d > 5:   return 'High'
        elif d > 1: return 'Medium'
        else:       return 'Low'

    df['risk_level'] = df['duration_mod'].apply(risk_level)

    return df