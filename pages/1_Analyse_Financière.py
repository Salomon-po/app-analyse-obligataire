# 1_Analyse_Financiere.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from calculs import preprocess_dataset
from config import COLORS, COLORS_RISK
from matplotlib.patches import Patch

# ─── CONFIGURATION UNIQUE DE LA PAGE ──────────────────────────────────────────
st.set_page_config(
    page_title = "Analyse Financière", 
    page_icon  = "images/Logo_Finance_Maxima_consulting.jpeg",
    layout     = "wide"
)

# ─── VÉRIFICATION DONNÉES ─────────────────────────────────────────────────────
if not st.session_state.get('data_prete', False):
    st.warning("Aucune donnée chargée — retournez à l'Accueil")
    st.stop()

# ─── CHARGEMENT ET PRÉPROCESSING ──────────────────────────────────────────────
st.title("Analyse Financière du Portefeuille")
df_raw = st.session_state['dataset']

# ─── CSS PERSONNALISÉ (STYLE SAAS PREMIUM INSTITUTIONNEL) ─────────────────────
st.markdown("""
<style>
    /* Style de la Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1F4E79;
    }
    /* Force le texte général de la sidebar en blanc sans casser les inputs */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {
        color: white !important;
    }
    
    /* Harmonisation des titres de la page principale */
    h1, h2, h3, h4 { 
        color: #1F4E79 !important; 
        font-weight: bold;
    }
    
    /* Bouton Principal Premium (Style Institutionnel) */
    .stButton > button {
        background-color: #1F4E79 !important;
        color: white !important;
        border: 1px solid #1F4E79 !important;
        border-radius: 6px !important;
        font-weight: bold !important;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #2E74B5 !important;
        border-color: #2E74B5 !important;
        color: white !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

with st.spinner("Calcul des métriques financières en cours..."):
    if 'dataset_processed' not in st.session_state:
        df = preprocess_dataset(df_raw)
        st.session_state['dataset_processed'] = df
    else:
        df = st.session_state['dataset_processed']

st.success(f"{len(df)} obligations analysées et prêtes pour l'évaluation")

# ─── MÉTRIQUES GLOBALES ───────────────────────────────────────────────────────
st.markdown("### Vue d'ensemble du portefeuille")
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Obligations",      len(df))
col2.metric("YTM moyen",        f"{df['ytm_calculee'].mean()*100:.2f}%")
col3.metric("Duration moyenne", f"{df['duration_mod'].mean():.2f} ans")
col4.metric("Convexité moy.",   f"{df['convexity'].mean():.2f}")
col5.metric("Actives",          f"{(df['statut']=='active').sum()}")

st.markdown("---")

# ─── TABLEAU DES RÉSULTATS ────────────────────────────────────────────────────
st.markdown("### Tableau des métriques par obligation")

cols_affichage = ['isin', 'type_obligation', 'Pays', 'Coupon',
                  'prix', 'ytm_calculee', 'duration_mac',
                  'duration_mod', 'convexity', 'sensibility',
                  'variation_prix', 'var_prix_sens_convex',
                  'risk_level', 'statut']

cols_disponibles = [c for c in cols_affichage if c in df.columns]
st.dataframe(df[cols_disponibles].round(4), use_container_width=True, hide_index=True)

st.markdown("---")

# ─── VISUALISATIONS (AVEC PARAMÈTRES GRAPHIQUES COHÉRENTS) ───────────────────
st.markdown("### Visualisations Graphiques")

# ── Paramètres matplotlib cohérents avec le design ────────────────────────────
plt.rcParams.update({
    'axes.facecolor':    '#FFFFFF',
    'figure.facecolor':  '#FFFFFF',
    'axes.edgecolor':    '#1F4E79',
    'axes.labelcolor':   '#1F4E79',
    'xtick.color':       '#1F4E79',
    'ytick.color':       '#1F4E79',
    'text.color':        '#1A1A2E',
    'grid.color':        '#EBF3FB',
    'axes.titlecolor':   '#1F4E79',
    'axes.grid':         True,         # Active la grille automatiquement sur la 2D
    'grid.linestyle':    '--',         # Style pointillé élégant
    'grid.linewidth':    0.7           # Lignes fines pour ne pas surcharger
})

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# ── Figure 1 : Distribution Risk Level ────────────────────────────────────────
risk_counts = df['risk_level'].value_counts()
colors_risk = [COLORS_RISK.get(r, '#808080') for r in risk_counts.index]
axes[0,0].bar(risk_counts.index, risk_counts.values, color=colors_risk, alpha=0.85, width=0.5)
axes[0,0].set_title('DISTRIBUTION DU RISK LEVEL', fontweight='bold')
axes[0,0].set_xlabel('Risk Level', fontsize=8)
axes[0,0].set_ylabel('Nombre d\'obligations', fontsize=8)
for i, v in enumerate(risk_counts.values):
    axes[0,0].text(i, v + (max(risk_counts.values)*0.01), str(v), ha='center', fontweight='bold')

# ── Figure 2 : YTM médian par type_obligation ─────────────────────────────────
if 'type_obligation' in df.columns:
    ytm_type = df.groupby('type_obligation')['ytm_calculee'].median().sort_values()
    axes[0,1].barh(ytm_type.index, ytm_type.values * 100, color=COLORS['primary'], alpha=0.85, height=0.5)
    axes[0,1].set_title("YTM MÉDIAN PAR TYPE D'OBLIGATION", fontweight='bold')
    axes[0,1].set_xlabel('YTM (%)', fontsize=8)

# ── Figure 3 : Distribution Duration Modifiée ─────────────────────────────────
axes[0,2].hist(df['duration_mod'], bins=30, color=COLORS['primary'], alpha=0.85, edgecolor='white')
axes[0,2].set_title('DISTRIBUTION DE LA DURATION MODIFIÉE', fontweight='bold')
axes[0,2].set_xlabel('Duration Modifiée', fontsize=8)
axes[0,2].set_ylabel('Fréquence', fontsize=8)

# ── Figure 4 : YTM vs Duration Modifiée ───────────────────────────────────────
df_plot = df[df['ytm_calculee'] < 0.20]  # ytm < 20% pour la visualisation
colors_scatter = [COLORS_RISK.get(r, '#808080') for r in df_plot['risk_level']]
axes[1,0].scatter(df_plot['duration_mod'], df_plot['ytm_calculee'] * 100, c=colors_scatter, alpha=0.6, s=40, zorder=5)
axes[1,0].set_title('YTM VS DURATION MODIFIÉE', fontweight='bold')
axes[1,0].set_xlabel('Duration Modifiée', fontsize=8)
axes[1,0].set_ylabel('YTM (%)', fontsize=8)

legend_elements = [Patch(facecolor=COLORS_RISK['Low'],    label='Low'),
                   Patch(facecolor=COLORS_RISK['Medium'], label='Medium'),
                   Patch(facecolor=COLORS_RISK['High'],   label='High')]
axes[1,0].legend(handles=legend_elements, fontsize=8, loc='upper right')

# ── Figure 5 : Top 10 pays ────────────────────────────────────────────────────
pays_counts = df['Pays'].value_counts().head(10)
axes[1,1].bar(pays_counts.index, pays_counts.values, color=COLORS['secondary'], alpha=0.85, width=0.5)
axes[1,1].set_title('DISTRIBUTION PAR PAYS (TOP 10)', fontweight='bold')
axes[1,1].set_xlabel('Pays', fontsize=8)
axes[1,1].set_ylabel('Nombre d\'obligations', fontsize=8)
axes[1,1].tick_params(axis='x', rotation=45, labelsize=8)

# ── Figure 6 : Boxplot Duration par Risk Level ────────────────────────────────
risk_order = ['Low', 'Medium', 'High']
data_box = [df[df['risk_level'] == r]['duration_mod'].dropna().values for r in risk_order]
bp = axes[1,2].boxplot(data_box, labels=risk_order, patch_artist=True)
for patch, color in zip(bp['boxes'], [COLORS_RISK['Low'], COLORS_RISK['Medium'], COLORS_RISK['High']]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('#1F4E79')
axes[1,2].set_title('DURATION MODIFIÉE PAR RISK LEVEL', fontweight='bold')
axes[1,2].set_xlabel('Risk Level', fontsize=8)
axes[1,2].set_ylabel('Duration Modifiée', fontsize=8)

# Ajustement automatique global pour supprimer tout espace vide résiduel
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# ─── STATISTIQUES DESCRIPTIVES ────────────────────────────────────────────────
st.markdown("### Statistiques descriptives globales")
cols_stats = ['ytm_calculee', 'duration_mac', 'duration_mod', 'convexity', 'sensibility', 'variation_prix']
cols_stats = [c for c in cols_stats if c in df.columns]
st.dataframe(df[cols_stats].describe().round(4), use_container_width=True, hide_index=False)