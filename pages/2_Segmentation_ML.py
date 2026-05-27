# 2_Segmentation_ML.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from config import COLORS, COLORS_RISK, FEATURES_ML, N_CLUSTERS, LINKAGE, METRIC_CLUST
from matplotlib.patches import Patch

# ─── CONFIGURATION UNIQUE DE LA PAGE ──────────────────────────────────────────
st.set_page_config(
    page_title = "Segmentation ML", 
    page_icon  = "images/Logo_Finance_Maxima_consulting.jpeg",
    layout     = "wide"
)

# ─── VÉRIFICATION DONNÉES ─────────────────────────────────────────────────────
if not st.session_state.get('data_prete', False):
    st.warning("Aucune donnée chargée — retournez à l'Accueil")
    st.stop()

if 'dataset_processed' not in st.session_state:
    st.warning("Veuillez d'abord passer par la Page 1 — Analyse Financière")
    st.stop()

# ─── CHARGEMENT DONNÉES ───────────────────────────────────────────────────────
st.title("Segmentation ML — Clustering Obligataire")
df = st.session_state['dataset_processed']

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

# ─── CLUSTERING ───────────────────────────────────────────────────────────────
with st.spinner("Clustering algorithmique en cours..."):
    if 'dataset_ml' not in st.session_state:

        # Features
        X = df[FEATURES_ML].copy()

        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Clustering
        model = AgglomerativeClustering(
            n_clusters = N_CLUSTERS,
            linkage    = LINKAGE,
            metric     = METRIC_CLUST
        )
        df['cluster'] = model.fit_predict(X_scaled)

        # Nommer les clusters selon duration
        profils = df.groupby('cluster')['duration_mod'].mean()
        sorted_clusters = profils.sort_values().index.tolist()
        cluster_map = {
            sorted_clusters[0] : 'Low',
            sorted_clusters[1] : 'Medium',
            sorted_clusters[2] : 'High',
        }
        df['risk_level_ml'] = df['cluster'].map(cluster_map)

        # Métriques
        silhouette = silhouette_score(X_scaled, df['cluster'])
        davies     = davies_bouldin_score(X_scaled, df['cluster'])
        calinski   = calinski_harabasz_score(X_scaled, df['cluster'])

        # Sauvegarder
        st.session_state['dataset_ml']    = df
        st.session_state['silhouette']    = silhouette
        st.session_state['davies']        = davies
        st.session_state['calinski']      = calinski
        st.session_state['X_scaled']      = X_scaled

    else:
        df         = st.session_state['dataset_ml']
        silhouette = st.session_state['silhouette']
        davies     = st.session_state['davies']
        calinski   = st.session_state['calinski']
        X_scaled   = st.session_state['X_scaled']

# ─── MÉTRIQUES DU MODÈLE ──────────────────────────────────────────────────────
st.markdown("### Métriques d'évaluation du modèle")
col1, col2, col3 = st.columns(3)
col1.metric("Silhouette Score",     f"{silhouette:.4f}", delta="Excellent > 0.7")
col2.metric("Davies-Bouldin",       f"{davies:.4f}", delta="Meilleur < 0.5", delta_color="inverse")
col3.metric("Calinski-Harabasz",    f"{calinski:.2f}", delta="Plus élevé = mieux")

st.markdown("---")

# ─── DISTRIBUTION DES CLUSTERS ────────────────────────────────────────────────
st.markdown("### Répartition et centres de gravité des clusters")
col_dist1, col_dist2 = st.columns([1, 2])

with col_dist1:
    dist = df['risk_level_ml'].value_counts()
    for risk, count in dist.items():
        pct = count / len(df) * 100
        st.metric(f"Cluster {risk}", f"{count} obligations", f"{pct:.1f}% de la base")

with col_dist2:
    st.dataframe(
        df.groupby('risk_level_ml')[FEATURES_ML].mean().round(4),
        use_container_width=True
    )

st.markdown("---")

# ─── TABLEAU DES OBLIGATIONS ──────────────────────────────────────────────────
st.markdown("### Liste détaillée des obligations segmentées")
cols_affichage = ['isin', 'type_obligation', 'Pays',
                  'ytm_calculee', 'duration_mod',
                  'convexity', 'risk_level', 'risk_level_ml']
cols_dispo = [c for c in cols_affichage if c in df.columns]
st.dataframe(df[cols_dispo].round(4), use_container_width=True, hide_index=True)

st.markdown("---")

# ─── VISUALISATIONS GRAPHIQUES ────────────────────────────────────────────────
st.markdown("### Analyses graphiques de la segmentation")

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

# Elements de légende standardisés
legend_elements = [Patch(facecolor=COLORS_RISK['Low'],    label='Low'),
                   Patch(facecolor=COLORS_RISK['Medium'], label='Medium'),
                   Patch(facecolor=COLORS_RISK['High'],   label='High')]

# ── Figure 1 : Distribution clusters ML
risk_counts = df['risk_level_ml'].value_counts()
colors_risk = [COLORS_RISK.get(r, '#808080') for r in risk_counts.index]
axes[0,0].bar(risk_counts.index, risk_counts.values, color=colors_risk, alpha=0.85, width=0.5)
axes[0,0].set_title('DISTRIBUTION DES CLUSTERS ML', fontweight='bold')
axes[0,0].set_xlabel('Risk Level ML', fontsize=8)
axes[0,0].set_ylabel("Nombre d'obligations", fontsize=8)
for i, v in enumerate(risk_counts.values):
    axes[0,0].text(i, v + (max(risk_counts.values)*0.01), str(v), ha='center', fontweight='bold')

# ── Figure 2 : YTM vs Duration par cluster
df_plot = df[df['ytm_calculee'] < 0.20]
colors_scatter = [COLORS_RISK.get(r, '#808080') for r in df_plot['risk_level_ml']]
axes[0,1].scatter(df_plot['duration_mod'], df_plot['ytm_calculee'] * 100, c=colors_scatter, alpha=0.6, s=40, zorder=5)
axes[0,1].set_title('YTM VS DURATION MODIFIÉE PAR CLUSTER', fontweight='bold')
axes[0,1].set_xlabel('Duration Modifiée', fontsize=8)
axes[0,1].set_ylabel('YTM (%)', fontsize=8)
axes[0,1].legend(handles=legend_elements, fontsize=8, loc='upper right')

# ── Figure 3 : Convexité vs Duration
axes[0,2].scatter(df_plot['duration_mod'], df_plot['convexity'], c=colors_scatter, alpha=0.6, s=40, zorder=5)
axes[0,2].set_title('CONVEXITÉ VS DURATION MODIFIÉE', fontweight='bold')
axes[0,2].set_xlabel('Duration Modifiée', fontsize=8)
axes[0,2].set_ylabel('Convexité', fontsize=8)
axes[0,2].legend(handles=legend_elements, fontsize=8, loc='upper left')

# ── Figure 4 : Heatmap comparaison Manuel vs ML
crosstab = pd.crosstab(df['risk_level'], df['risk_level_ml'])
sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues', ax=axes[1,0], cbar=False, annot_kws={'fontweight': 'bold'})
axes[1,0].set_title('MATRICE DE CONFUSION : MANUEL VS ML', fontweight='bold')
axes[1,0].set_xlabel('Risk Level ML', fontsize=8)
axes[1,0].set_ylabel('Risk Level Manuel', fontsize=8)

# ── Figure 5 : Boxplot YTM par cluster
risk_order = ['Low', 'Medium', 'High']
df_filtre = df[df['ytm_calculee'] < 0.20]
data_box = [df_filtre[df_filtre['risk_level_ml'] == r]['ytm_calculee'].dropna().values * 100 for r in risk_order]
bp = axes[1,1].boxplot(data_box, labels=risk_order, patch_artist=True)
for patch, color in zip(bp['boxes'], [COLORS_RISK['Low'], COLORS_RISK['Medium'], COLORS_RISK['High']]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('#1F4E79')
axes[1,1].set_title('PROFIL RENDEMENT (YTM) PAR CLUSTER', fontweight='bold')
axes[1,1].set_xlabel('Risk Level ML', fontsize=8)
axes[1,1].set_ylabel('YTM (%)', fontsize=8)

# ── Figure 6 : Boxplot Duration par cluster
data_box2 = [df[df['risk_level_ml'] == r]['duration_mod'].dropna().values for r in risk_order]
bp2 = axes[1,2].boxplot(data_box2, labels=risk_order, patch_artist=True)
for patch, color in zip(bp2['boxes'], [COLORS_RISK['Low'], COLORS_RISK['Medium'], COLORS_RISK['High']]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('#1F4E79')
axes[1,2].set_title('PROFIL DE RISK SENSITIVITY (DURATION)', fontweight='bold')
axes[1,2].set_xlabel('Risk Level ML', fontsize=8)
axes[1,2].set_ylabel('Duration Modifiée', fontsize=8)

# Nettoyage automatique des marges blanches inter-graphes
plt.tight_layout()
st.pyplot(fig)

# ─── MISE À TRACE DE SESSION STATE ────────────────────────────────────────────
st.session_state['dataset_processed'] = df
st.session_state['dataset_ml']        = df