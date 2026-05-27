# 5_Optimisation.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize
from config import COLORS, COLORS_RISK, POIDS_MAX, DURATION_MIN

st.set_page_config(
    page_title = "Optimisation",
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
st.title("Optimisation & Arbitrage du Portefeuille")
df = st.session_state['dataset_ml']

# ─── CSS PERSONNALISÉ (CORRIGÉ POUR COHÉRENCE SLIDERS ET BOUTONS) ──────────────
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

# ─── PARAMÈTRES OPTIMISATION (DANS LE CORPS PRINCIPAL) ────────────────────────
st.header("Paramètres de calcul")
col_param1, col_param2, col_param3 = st.columns(3)

with col_param1:
    poids_max = st.slider(
        "Poids maximum / obligation (%)",
        min_value=1, max_value=20,
        value=int(POIDS_MAX * 100), step=1
    ) / 100

with col_param2:
    duration_min = st.slider(
        "Duration minimale (ans)",
        min_value=0.1, max_value=5.0,
        value=float(DURATION_MIN), step=0.1
    )

with col_param3:
    rf_global = st.number_input(
        "Taux sans risque global (%)",
        min_value=0.0, max_value=10.0,
        value=2.97, step=0.1
    ) / 100

# ─── PRÉPARATION DONNÉES ──────────────────────────────────────────────────────
mask = (
    (df['statut'] == 'active') &
    (df['ytm_calculee'] < 0.20) &
    (df['duration_mod'] > 0.1) &
    (df['prix'] > 10)
)
df_opt = df[mask].copy().reset_index(drop=True)

st.info(f"{len(df_opt)} obligations valides sélectionnées pour l'optimisation.")

mu    = df_opt['ytm_calculee'].values
dur   = df_opt['duration_mod'].values
n     = len(mu)

if n == 0:
    st.error("Aucune obligation ne respecte les critères de filtrage.")
    st.stop()

w0    = np.ones(n) / n
rf    = rf_global

# Matrice de covariance simplifiée
sigma_i = dur / dur.max() if dur.max() > 0 else np.ones(n)
Sigma   = np.outer(sigma_i, sigma_i) * 0.1 + np.diag(sigma_i**2)

# Fonctions financières
def rendement(w):  return np.dot(w, mu)
def risque(w):     return np.sqrt(np.dot(w.T, np.dot(Sigma, w)))
def duration_p(w): return np.dot(w, dur)
def sharpe(w):
    r = risque(w)
    return (rendement(w) - rf) / r if r > 0 else 0

poids_max_ajuste = max(poids_max, 1.0 / n)
bounds           = [(0, poids_max_ajuste)] * n
constraints_base = [
    {'type': 'eq',   'fun': lambda w: np.sum(w) - 1},
    {'type': 'ineq', 'fun': lambda w: duration_p(w) - duration_min},
]

# ─── AFFICHAGE : PORTFEUILLE ACTUEL ───────────────────────────────────────────
st.subheader("État Actuel du Portefeuille")
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
m_col1.metric("Rendement (YTM)",  f"{rendement(w0)*100:.2f}%")
m_col2.metric("Risque Volatilité", f"{risque(w0)*100:.4f}%")
m_col3.metric("Duration Moyenne",   f"{duration_p(w0):.2f} ans")
m_col4.metric("Sharpe Ratio",     f"{sharpe(w0):.4f}")

st.markdown("---")

# ─── ACTION OPTIMISATION ──────────────────────────────────────────────────────
if st.button("Lancer l'Analyse & l'Arbitrage des Modèles", type="primary", use_container_width=True):
    with st.spinner("Calcul et ajustement de la structure visuelle..."):

        # ── 1. CALCULS FRONTIÈRE DE MARKOWITZ ─────────────────────────────────
        r_min, r_max = mu.min(), mu.max()
        rendements_cibles = np.linspace(r_min + (r_max-r_min)*0.01, r_max - (r_max-r_min)*0.01, 20) if r_min != r_max else [r_min]
        risques_eff, rendements_eff, weights_eff = [], [], []

        for R_cible in rendements_cibles:
            constraints_eff = constraints_base + [{'type': 'eq', 'fun': lambda w, R=R_cible: rendement(w) - R}]
            result = minimize(risque, w0, method='SLSQP', bounds=bounds, constraints=constraints_eff)
            if result.success:
                risques_eff.append(result.fun)
                rendements_eff.append(R_cible)
                weights_eff.append(result.x)
                
        if len(risques_eff) > 0:
            idx_min = np.argmin(risques_eff)
            w_min_var = weights_eff[idx_min]
        else:
            st.error("L'optimisation a échoué (Contraintes insolvables).")
            st.stop()
        
        # ── 2. CALCULS DES AUTRES STRATÉGIES ──────────────────────────────────
        result_sharpe = minimize(lambda w: -sharpe(w), w0, method='SLSQP', bounds=bounds, constraints=constraints_base + [{'type': 'ineq', 'fun': lambda w: rendement(w) - rf}])
        w_sharpe = result_sharpe.x if result_sharpe.success else w0

        result_dur = minimize(duration_p, w0, method='SLSQP', bounds=bounds, constraints=constraints_base + [{'type': 'ineq', 'fun': lambda w: rendement(w) - rendement(w0)}])
        w_dur = result_dur.x if result_dur.success else w0

        # ── 3. DATAFRAME ET ALGORITHME D'ARBITRAGE AUTOMATIQUE ────────────────
        all_w = [w0, w_min_var, w_sharpe, w_dur]
        noms_portefeuilles = ['Portefeuille Actuel', 'Variance Minimale', 'Sharpe Optimal', 'Duration Minimale']
        
        r_vals = np.array([rendement(w) for w in all_w])
        std_vals = np.array([risque(w) for w in all_w])
        dur_vals = np.array([duration_p(w) for w in all_w])
        sh_vals = np.array([sharpe(w) for w in all_w])

        def norm_max(x): return (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else np.ones_like(x)
        def norm_min(x): return (x.max() - x) / (x.max() - x.min()) if x.max() != x.min() else np.ones_like(x)

        score_rendement = norm_max(r_vals)
        score_sharpe    = norm_max(sh_vals)
        score_risque    = norm_min(std_vals)
        score_duration  = norm_min(dur_vals)

        scores_globaux = (score_rendement + score_sharpe + score_risque + score_duration) * 25
        idx_best = np.argmax(scores_globaux)
        meilleur_modele = noms_portefeuilles[idx_best]

        # ── 4. PANNEAU DE RÉSULTATS ET VERDICT ────────────────────────────────
        st.subheader("Tableau d'Arbitrage et Notation des Optimiseurs")
        comparatif = pd.DataFrame({
            'Modèle de Portefeuille' : noms_portefeuilles,
            'Rendement'  : [f"{r*100:.2f}%" for r in r_vals],
            'Risque (Volatilité)' : [f"{v*100:.3f}%" for v in std_vals],
            'Duration (ans)'     : [f"{d:.2f}" for d in dur_vals],
            'Sharpe Ratio'       : [f"{s:.3f}" for s in sh_vals],
            'Score de Synthèse'  : [f"{score:.1f} / 100" for score in scores_globaux]
        })
        
        st.dataframe(comparatif, use_container_width=True, hide_index=True)

        st.success(f"**Le meilleur choix mathématique calculé est : {meilleur_modele}** ({scores_globaux[idx_best]:.1f}/100)")

        # ── 5. GENERATION DE LA GRILLE UNIQUE (CORRIGÉE ET AJUSTÉE) ───────────
        st.subheader("Structure de Visualisation d'Origine")
        
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
            'axes.grid':         True,         # Active la grille sur les graphiques 2D
            'grid.linestyle':    '--',         # Donne un style pointillé élégant
            'grid.linewidth':    0.7           # Lignes fines pour ne pas surcharger
        })

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        portefeuilles_label = ['Actuel', 'Var. Min.', 'Sharpe Opt.', 'Duration Min.']

        # ── GRAPHIQUE [0, 0] : Frontière Efficiente
        if len(risques_eff) > 1:
            axes[0, 0].plot([r * 100 for r in risques_eff], [m * 100 for m in rendements_eff], color='#1F4E79', linewidth=2, label='Frontière')
        axes[0, 0].scatter(std_vals[0]*100, r_vals[0]*100, color='black', s=100, zorder=5, label='Actuel')
        axes[0, 0].scatter(std_vals[1]*100, r_vals[1]*100, color='#28A745', s=100, zorder=5, label='Var. Min.')
        axes[0, 0].scatter(std_vals[2]*100, r_vals[2]*100, color='#DC3545', s=100, zorder=5, label='Sharpe Opt.')
        axes[0, 0].scatter(std_vals[3]*100, r_vals[3]*100, color='purple', s=100, zorder=5, label='Duration Min.')
        axes[0, 0].axhline(y=rf*100, color='#FFC107', linestyle='--')
        axes[0, 0].set_title('MARKOWITZ EFFICIENT FRONTIER', fontsize=10, fontweight='bold')
        axes[0, 0].set_xlabel('Risque (Volatilité %)', fontsize=8)
        axes[0, 0].set_ylabel('Rendement (%)', fontsize=8)
        axes[0, 0].legend(fontsize=7, loc='lower right')

        # ── GRAPHIQUE [0, 1] : Arbitrage Rendement / Risque
        colors_p = ['black', '#28A745', '#DC3545', 'purple']
        for p, r, ri, c in zip(portefeuilles_label, r_vals*100, std_vals*100, colors_p):
            axes[0, 1].scatter(ri, r, color=c, s=100, zorder=5)
            axes[0, 1].annotate(p, (ri, r), textcoords="offset points", xytext=(4, 4), fontsize=7, fontweight='bold')
        axes[0, 1].set_title('RISK & RETURN TRADE-OFF', fontsize=10, fontweight='bold')
        axes[0, 1].set_xlabel('Risque (Volatilité %)', fontsize=8)
        axes[0, 1].set_ylabel('Rendement (%)', fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)

        # ── GRAPHIQUE [0, 2] : Sharpe Ratio Comparison
        axes[0, 2].bar(portefeuilles_label, sh_vals, color=colors_p, alpha=0.8, width=0.45)
        axes[0, 2].axhline(y=0, color='black', linewidth=0.8)
        axes[0, 2].set_title('SHARPE RATIO COMPARISON', fontsize=10, fontweight='bold')
        axes[0, 2].grid(axis='y', alpha=0.3)
        for i, s in enumerate(sh_vals):
            axes[0, 2].text(i, s + (0.02 if s >= 0 else -0.06), f'{s:.2f}', ha='center', fontsize=8, fontweight='bold')

        # ── GRAPHIQUE [1, 0] : Duration Moyenne
        axes[1, 0].bar(portefeuilles_label, dur_vals, color=colors_p, alpha=0.8, width=0.45)
        axes[1, 0].set_title('PORTFOLIO DURATION (YEARS)', fontsize=10, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        for i, d in enumerate(dur_vals):
            axes[1, 0].text(i, d + 0.05, f'{d:.2f}', ha='center', fontsize=8, fontweight='bold')

        # ── GRAPHIQUE [1, 1] : Radar de Diagnostic Multi-Critères
        pos = axes[1, 1].get_position()
        axes[1, 1].remove()
        ax_radar = fig.add_axes(pos, polar=True)

        categories = ['Rendement', 'Sharpe (norm.)', 'Sécurité Risque', 'Maîtrise Duration']
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        for i, label in enumerate(portefeuilles_label):
            values = [score_rendement[i], score_sharpe[i], score_risque[i], score_duration[i]]
            values += values[:1]
            ax_radar.plot(angles, values, color=colors_p[i], linewidth=1.5, label=label)
            ax_radar.fill(angles, values, color=colors_p[i], alpha=0.02)

        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories, fontsize=7, fontweight='bold')
        ax_radar.set_title('GLOBAL PORTFOLIO DIAGNOSTIC', fontsize=10, fontweight='bold', pad=10)
        ax_radar.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=2, fontsize=7)

        # ── GRAPHIQUE [1, 2] : Allocation / Répartition Cible (Top Sharpe)
        df_opt['poids'] = w_sharpe
        top10 = df_opt[df_opt['poids'] > 0.001].nlargest(min(10, len(df_opt)), 'poids')
        colors_pays_list = {'USA': '#1F4E79', 'DE': '#F0C040', 'Germany': '#F0C040', 'France': '#2E74B5', 'UK': '#C00000', 'United Kingdom': '#C00000', 'Japon': '#E07040', 'Japan': '#E07040'}
        bar_colors = [colors_pays_list.get(p, '#808080') for p in top10['Pays']]
        nom_col_type = 'type' if 'type' in top10.columns else 'type_obligation'

        axes[1, 2].barh(range(len(top10)), top10['poids']*100, color=bar_colors, alpha=0.85)
        axes[1, 2].set_yticks(range(len(top10)))
        axes[1, 2].set_yticklabels([f"{row[nom_col_type]} ({row['Pays']})" for _, row in top10.iterrows()], fontsize=7)
        axes[1, 2].set_title('TOP ALLOCATION (SHARPE)', fontsize=10, fontweight='bold')
        axes[1, 2].set_xlabel('Poids (%)', fontsize=8)
        axes[1, 2].grid(axis='x', alpha=0.3)

        # Suppression des marges pour un rendu ultra-serré
        plt.tight_layout()
        st.pyplot(fig)

        # ── Sauvegardes Session State
        st.session_state['w_sharpe']   = w_sharpe
        st.session_state['w_dur']      = w_dur
        st.session_state['w_min_var']  = w_min_var
        st.session_state['comparatif'] = comparatif