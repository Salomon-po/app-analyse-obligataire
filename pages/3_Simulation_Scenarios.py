# 4_Simulation_Scenarios.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import COLORS, COLORS_RISK, COLORS_PAYS, SCENARIOS, BANQUES_CENTRALES, ANTICIPATIONS_BC_DEFAUT

# ─── CONFIGURATION UNIQUE DE LA PAGE ──────────────────────────────────────────
st.set_page_config(
    page_title = "Simulation Scénarios",
    page_icon  = "images/Logo_Finance_Maxima_consulting.jpeg",
    layout     = "wide"
)

# ─── VÉRIFICATION DONNÉES ─────────────────────────────────────────────────────
if not st.session_state.get('data_prete', False):
    st.warning("Aucune donnée chargée — retournez à l'Accueil")
    st.stop()

if 'dataset_ml' not in st.session_state:
    st.warning("Veuillez d'abord passer par les Pages 1 et 2")
    st.stop()

# ─── CHARGEMENT ───────────────────────────────────────────────────────────────
st.title("Simulation de Scénarios de Stress-Testing")
df = st.session_state['dataset_ml']

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
    
    /* Bouton et sélections */
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

# ─── CHOIX DU SCÉNARIO ────────────────────────────────────────────────────────
st.markdown("### Configuration du choc macroéconomique")
col_scen1, col_scen2 = st.columns([1, 2])

with col_scen1:
    scenario_choisi = st.radio(
        "Sélectionner un scénario de marché",
        options=list(SCENARIOS.keys()) + ['Personnalisé'],
        index=0
    )

with col_scen2:
    if scenario_choisi == 'Personnalisé':
        st.markdown("**Définissez vos chocs de taux par pays (%)**")
        chocs_custom = {}
        for pays, bc in BANQUES_CENTRALES.items():
            choc = st.slider(
                f"{bc} — {pays}",
                min_value = -2.0,
                max_value =  2.0,
                value     = float(ANTICIPATIONS_BC_DEFAUT[pays] * 100),
                step      = 0.25,
                key       = f"choc_{pays}"
            )
            chocs_custom[pays] = choc / 100
        chocs_par_pays = chocs_custom
    else:
        chocs_par_pays = SCENARIOS[scenario_choisi]
        st.markdown("**Chocs appliqués par zone monétaire :**")
        df_chocs = pd.DataFrame({
            'Pays'           : list(chocs_par_pays.keys()),
            'Banque Centrale': [BANQUES_CENTRALES[p] for p in chocs_par_pays.keys()],
            'Choc (%)'       : [f"{v*100:.2f}%" for v in chocs_par_pays.values()],
            'Direction'      : ['↑ Hausse des taux' if v > 0 else '↓ Baisse des taux' if v < 0 else '→ Neutre'
                                for v in chocs_par_pays.values()]
        })
        st.dataframe(df_chocs, use_container_width=True, hide_index=True)

st.markdown("---")

# ─── CALCUL IMPACT ────────────────────────────────────────────────────────────
df['choc_pays'] = df['Pays'].map(chocs_par_pays).fillna(0.0)

# Alerte pays manquants
pays_sans_choc = df[df['choc_pays'].isna()]['Pays'].unique()
if len(pays_sans_choc) > 0:
    st.warning(f"Pays sans choc défini (0% appliqué) : {list(pays_sans_choc)}")

# Formule d'approximation par Taylor (Durée modifiée + Convexité)
df['variation_prix_future'] = (
    -df['duration_mod'] * df['choc_pays']
    + 0.5 * df['convexity'] * df['choc_pays']**2
)

# ─── MÉTRIQUES GLOBALES IMPACT ────────────────────────────────────────────────
st.markdown("### Impact consolidé sur le portefeuille")
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

m_col1.metric("Impact moyen du portefeuille",
              f"{df['variation_prix_future'].mean()*100:.2f}%",
              delta=f"{df['variation_prix_future'].mean()*100:.2f}%",
              delta_color="inverse" if df['variation_prix_future'].mean() < 0 else "normal")
m_col2.metric("Pire performance (Max Drawdown)", f"{df['variation_prix_future'].min()*100:.2f}%")
m_col3.metric("Meilleure performance", f"{df['variation_prix_future'].max()*100:.2f}%")
m_col4.metric("Nombre d'obligations en perte", f"{(df['variation_prix_future'] < 0).sum()}")

st.markdown("---")

# ─── TOUS LES SCÉNARIOS COMPARATIF ────────────────────────────────────────────
st.markdown("### Analyse comparative inter-scénarios")

resultats_scenarios = {}
for scenario, chocs in SCENARIOS.items():
    df[f'var_{scenario}'] = (
        -df['duration_mod'] * df['Pays'].map(chocs)
        + 0.5 * df['convexity'] * df['Pays'].map(chocs)**2
    )
    resultats_scenarios[scenario] = {
        'Impact Moyen'    : df[f'var_{scenario}'].mean() * 100,
        'Impact High Risk': df[df['risk_level_ml']=='High'][f'var_{scenario}'].mean() * 100,
        'Impact Med Risk' : df[df['risk_level_ml']=='Medium'][f'var_{scenario}'].mean() * 100,
        'Impact Low Risk' : df[df['risk_level_ml']=='Low'][f'var_{scenario}'].mean() * 100,
        'Pire Obligation' : df[f'var_{scenario}'].min() * 100,
        'Top Obligation'  : df[f'var_{scenario}'].max() * 100,
    }

df_scenarios = pd.DataFrame(resultats_scenarios).T.round(4)
st.dataframe(df_scenarios, use_container_width=True, hide_index=False)

st.markdown("---")

# ─── VISUALISATIONS GRAPHIQUES ────────────────────────────────────────────────
st.markdown("### Graphiques d'impacts et de sensibilité")

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
    'axes.grid':         True,
    'grid.linestyle':    '--',
    'grid.linewidth':    0.7
})

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
scenarios_names = list(SCENARIOS.keys())

# ── Figure 1 : Impact moyen par scénario
impacts_moyens = [resultats_scenarios[s]['Impact Moyen'] for s in scenarios_names]
colors_bar = ['#375623' if x > 0 else '#C00000' for x in impacts_moyens]
axes[0,0].bar(scenarios_names, impacts_moyens, color=colors_bar, alpha=0.85, width=0.45)
axes[0,0].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
axes[0,0].set_title('IMPACT MOYEN PAR SCÉNARIO (%)', fontweight='bold')
axes[0,0].set_ylabel('Variation de Prix (%)', fontsize=8)
axes[0,0].tick_params(axis='x', rotation=20, labelsize=8)

# ── Figure 2 : Impact par cluster et scénario
x = np.arange(len(scenarios_names))
width = 0.22
high_imp   = [resultats_scenarios[s]['Impact High Risk'] for s in scenarios_names]
medium_imp = [resultats_scenarios[s]['Impact Med Risk']  for s in scenarios_names]
low_imp    = [resultats_scenarios[s]['Impact Low Risk']  for s in scenarios_names]

axes[0,1].bar(x - width, high_imp,   width, label='High',   color=COLORS_RISK['High'],   alpha=0.85)
axes[0,1].bar(x,         medium_imp, width, label='Medium', color=COLORS_RISK['Medium'], alpha=0.85)
axes[0,1].bar(x + width, low_imp,    width, label='Low',    color=COLORS_RISK['Low'],    alpha=0.85)
axes[0,1].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
axes[0,1].set_title('IMPACT PAR NIVEAU DE RISQUE ML (%)', fontweight='bold')
axes[0,1].set_xticks(x)
axes[0,1].set_xticklabels(scenarios_names, rotation=20, fontsize=8)
axes[0,1].legend(fontsize=8)

# ── Figure 3 : Pire et meilleure obligation
pires      = [resultats_scenarios[s]['Pire Obligation'] for s in scenarios_names]
meilleures = [resultats_scenarios[s]['Top Obligation']  for s in scenarios_names]
axes[0,2].bar(x - 0.18, meilleures, 0.3, label='Meilleure', color='#28A745', alpha=0.85)
axes[0,2].bar(x + 0.18, pires,      0.3, label='Pire',      color='#DC3545', alpha=0.85)
axes[0,2].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
axes[0,2].set_title('DISPERSION EXTRÊME PAR SCÉNARIO (%)', fontweight='bold')
axes[0,2].set_xticks(x)
axes[0,2].set_xticklabels(scenarios_names, rotation=20, fontsize=8)
axes[0,2].legend(fontsize=8)

# ── Figure 4 : Impact par pays — scénario sélectionné
impact_pays = df.groupby('Pays')['variation_prix_future'].mean() * 100
colors_pays_bar = ['#375623' if x > 0 else '#C00000' for x in impact_pays.values]
axes[1,0].bar(impact_pays.index, impact_pays.values, color=colors_pays_bar, alpha=0.85, width=0.45)
axes[1,0].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
axes[1,0].set_title(f'IMPACT RENDEMENT PAR PAYS — {scenario_choisi.upper()}', fontweight='bold')
axes[1,0].set_ylabel('Variation de Prix (%)', fontsize=8)
axes[1,0].tick_params(axis='x', rotation=30, labelsize=8)

# ── Figure 5 : Heatmap pays × scénario
heatmap_data = {}
for scenario, chocs in SCENARIOS.items():
    heatmap_data[scenario] = df.groupby('Pays')[f'var_{scenario}'].mean() * 100

heatmap_df = pd.DataFrame(heatmap_data)
sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=axes[1,1], cbar=False, annot_kws={'fontweight':'bold','size':8})
axes[1,1].set_title('HEATMAP GLOBAL IMPACT : PAYS × SCÉNARIO', fontweight='bold')
axes[1,1].tick_params(axis='x', rotation=20, labelsize=8)

# ── Figure 6 : Courbe impact convexe théorique vs choc
chocs_range = np.linspace(-0.02, 0.02, 100)
dur_high    = float(df[df['risk_level_ml']=='High']['duration_mod'].mean())
dur_medium  = float(df[df['risk_level_ml']=='Medium']['duration_mod'].mean())
dur_low     = float(df[df['risk_level_ml']=='Low']['duration_mod'].mean())
conv_high   = float(df[df['risk_level_ml']=='High']['convexity'].mean())
conv_medium = float(df[df['risk_level_ml']=='Medium']['convexity'].mean())
conv_low    = float(df[df['risk_level_ml']=='Low']['convexity'].mean())

imp_high   = [(-dur_high   * c + 0.5 * conv_high   * c**2) * 100 for c in chocs_range]
imp_medium = [(-dur_medium * c + 0.5 * conv_medium * c**2) * 100 for c in chocs_range]
imp_low    = [(-dur_low    * c + 0.5 * conv_low    * c**2) * 100 for c in chocs_range]

axes[1,2].plot(chocs_range*100, imp_high,   label='High',   color=COLORS_RISK['High'],   linewidth=2.2)
axes[1,2].plot(chocs_range*100, imp_medium, label='Medium', color=COLORS_RISK['Medium'], linewidth=2.2)
axes[1,2].plot(chocs_range*100, imp_low,    label='Low',    color=COLORS_RISK['Low'],    linewidth=2.2)
axes[1,2].axhline(y=0, color='black', linestyle='--', alpha=0.4, linewidth=0.8)
axes[1,2].axvline(x=0, color='black', linestyle='--', alpha=0.4, linewidth=0.8)
axes[1,2].set_title('SIMULATION THÉORIQUE : COUPE SENSITIVITÉ', fontweight='bold')
axes[1,2].set_xlabel('Choc de taux (Points de Pourcentage)', fontsize=8)
axes[1,2].set_ylabel('Variation de Prix (%)', fontsize=8)
axes[1,2].legend(fontsize=8)

# Équilibrage parfait des subplots
plt.tight_layout()
st.pyplot(fig)

# ─── TOP 10 OBLIGATIONS IMPACTÉES ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### Focus sur les 10 obligations les plus affectées négativement")
cols_top = ['isin', 'type_obligation', 'Pays', 'duration_mod',
            'choc_pays', 'variation_prix_future', 'risk_level_ml']
cols_dispo = [c for c in cols_top if c in df.columns]
top10 = df[cols_dispo].nsmallest(10, 'variation_prix_future')

# Modification de l'affichage en pourcentages lisibles avant rendu table
if 'choc_pays' in top10.columns:
    top10['choc_pays'] = top10['choc_pays'].apply(lambda val: f"{val*100:.2f}%")
if 'variation_prix_future' in top10.columns:
    top10['variation_prix_future'] = top10['variation_prix_future'].apply(lambda val: f"{val*100:.2f}%")

st.dataframe(top10, use_container_width=True, hide_index=True)

# ─── MISE À JOUR SESSION STATE ────────────────────────────────────────────────
st.session_state['dataset_ml']       = df
st.session_state['chocs_par_pays']   = chocs_par_pays