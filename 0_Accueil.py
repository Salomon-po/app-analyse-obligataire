import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from config import (
    PAYS_SUPPORTES, TYPES_OBLIGATIONS, DEVISES,
    COLS_OBLIGATOIRES, COLS_OPTIONNELLES,
    COLORS, ADMIN_PASSWORD, ANTICIPATIONS_BC_DEFAUT,
    BANQUES_CENTRALES
)

# ─── CONFIGURATION PAGE ───────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Outil Obligataire",
    page_icon  = "images/Logo_FInance_Maxima_consulting.jpeg",
    layout     = "wide",
)

# ─── CSS PERSONNALISÉ ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #1F4E79;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    .main-header {
        background-color: #1F4E79;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #EBF3FB;
        border-left: 4px solid #1F4E79;
        padding: 15px;
        border-radius: 5px;
    }
    h1, h2, h3 { color: #1F4E79; }
    .stButton > button {
        background-color: #1F4E79;
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #2E74B5;
    }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1> Outil d'Aide à la Décision</h1>
    <p>Gestion Obligataire — Analyse · Simulation · Optimisation</p>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
st.sidebar.image("images/Logo_FInance_Maxima_consulting.jpeg", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.markdown("### Données")

# ─── INITIALISATION SESSION STATE ─────────────────────────────────────────────
if 'dataset' not in st.session_state:
    st.session_state['dataset']       = None
if 'data_prete' not in st.session_state:
    st.session_state['data_prete']    = False
if 'chocs_par_pays' not in st.session_state:
    st.session_state['chocs_par_pays'] = ANTICIPATIONS_BC_DEFAUT.copy()

# ─── CHOIX MODE DE SAISIE ─────────────────────────────────────────────────────
mode = st.radio(
    "Comment souhaitez-vous entrer vos données ?",
    options=["Importer un fichier CSV", "Saisie manuelle"],
    horizontal=True
)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# FONCTION DE NORMALISATION 
# ══════════════════════════════════════════════════════════════════════════════
def normaliser_colonnes(df):
    """
    Normalise les noms de colonnes :
     supprime les espaces
     gere majuscules/minuscules
    """
    # Mapping des variantes possibles fourni
    mapping = {
        'type_obligation' : ['type_obligation', 'Type_Obligation', 'TYPE_OBLIGATION', 'type obligation', 
                     'TypeObligation', 'type', 'Type', 'TYPE', 'asset_class', 'AssetClass', 
                     'kind', 'category', 'catégorie', 'Categorie', 'secteur', 'Secteur'],
        'date_emission'   : ['date_emission', 'Date_Emission', 'DATE_EMISSION', 'dateemission', 'date emission'],
        'isin'            : ['isin', 'ISIN', 'Isin', 'code_isin', 'CODE_ISIN'],
        'Coupon'          : ['coupon', 'Coupon', 'COUPON', 'taux_coupon', 'taux coupon'],
        'date_maturite'   : ['date_maturite', 'Date_Maturite', 'DATE_MATURITE', 'maturite', 'date maturite', 'maturity_date'],
        'ytm'             : ['ytm', 'YTM', 'Ytm', 'rendement', 'yield'],
        'prix'            : ['prix', 'Prix', 'PRIX', 'price', 'Price', 'PRICE'],
        'benchmark'       : ['benchmark', 'Benchmark', 'BENCHMARK', 'taux_ref'],
        'Pays'            : ['pays', 'Pays', 'PAYS', 'country', 'Country', 'COUNTRY'],
        'devise'          : ['devise', 'Devise', 'DEVISE', 'currency', 'Currency', 'CURRENCY'],
    }

    # Renommer les colonnes trouvées
    rename_dict = {}
    cols_lower = {c.lower().strip(): c for c in df.columns}

    for col_cible, variantes in mapping.items():
        for variante in variantes:
            if variante.lower().strip() in cols_lower:
                col_originale = cols_lower[variante.lower().strip()]
                if col_originale != col_cible:
                    rename_dict[col_originale] = col_cible
                break

    df = df.rename(columns=rename_dict)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — IMPORT CSV
# ══════════════════════════════════════════════════════════════════════════════
if mode == "Importer un fichier CSV":
    st.subheader("Import du fichier CSV")
    uploaded_file = st.file_uploader(
        "Choisissez votre fichier CSV",
        type=['csv'],
        help="Colonnes requises : isin, Coupon, date_maturite, prix, Pays"
    )
    if uploaded_file:
        # ── Lecture avec détection séparateur ────────────────────────────────
        try:
            df = pd.read_csv(uploaded_file, sep=',')
            df = normaliser_colonnes(df)  
            
            if df.shape[1] == 1:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=';')
                df = normaliser_colonnes(df)  
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=';')
            df = normaliser_colonnes(df)  

        st.success(f"Fichier chargé : {len(df)} obligations détectées")

        # ── Validation colonnes obligatoires ──────────────────────────────────
        # (La validation s'exécute maintenant sur les colonnes nettoyées !)
        cols_manquantes = [c for c in COLS_OBLIGATOIRES
                           if c not in df.columns]
        if cols_manquantes:
            st.error(f"Colonnes manquantes : {cols_manquantes}")
            st.stop()

        # ── Colonnes optionnelles manquantes → suggestions ────────────────────
        st.markdown("### Colonnes détectées")
        for col, default in COLS_OPTIONNELLES.items():
            if col not in df.columns:
                st.warning(f" '{col}' absente → valeur suggérée : **{default}**")
                valeur = st.text_input(
                    f"Valeur pour '{col}'",
                    value=str(default) if default else "",
                    key=f"input_{col}"
                )
                df[col] = valeur if valeur else default

        # ── Détection format Coupon ───────────────────────────────────────────
        if df['Coupon'].max() > 1:
            st.info("Coupon détecté en pourcentage → converti en décimal")
            df['Coupon'] = df['Coupon'] / 100

        # ── Aperçu dataset ────────────────────────────────────────────────────
        st.markdown("### Aperçu du dataset")
        st.dataframe(df.head(10), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Obligations", len(df))
        col2.metric("Pays", df['Pays'].nunique())
        col3.metric("Types", df['type_obligation'].nunique()
                    if 'type_obligation' in df.columns else "N/A")

        # ── Sauvegarder en session state ──────────────────────────────────────
        if st.button("Lancer l'analyse", type="primary"):
            st.session_state['dataset']    = df
            st.session_state['data_prete'] = True
            st.success("Données prêtes ! Naviguez vers les pages d'analyse.")
            st.balloons()

# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — SAISIE MANUELLE
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.subheader("Saisie manuelle des obligations")

    # Initialiser liste obligations
    if 'obligations_manuelles' not in st.session_state:
        st.session_state['obligations_manuelles'] = []

    # ── Formulaire de saisie ──────────────────────────────────────────────────
    with st.form("form_obligation", clear_on_submit=True):
        st.markdown("#### Ajouter une obligation")

        col1, col2, col3 = st.columns(3)

        with col1:
            isin             = st.text_input("ISIN", placeholder="ex: US912828ZT51")
            type_obligation  = st.selectbox("Type", TYPES_OBLIGATIONS)
            pays             = st.selectbox("Pays", PAYS_SUPPORTES)

        with col2:
            coupon           = st.number_input("Coupon (%)", min_value=0.0,
                                               max_value=30.0, value=5.0, step=0.25)
            prix             = st.number_input("Prix", min_value=0.01,
                                               max_value=500.0, value=100.0, step=0.1)
            devise           = st.selectbox("Devise", DEVISES)

        with col3:
            date_emission    = st.date_input("Date émission",
                                             value=datetime(2020, 1, 1))
            date_maturite    = st.date_input("Date maturité",
                                             value=datetime(2030, 1, 1))
            ytm              = st.number_input("YTM original (%)", min_value=0.0,
                                               max_value=100.0, value=3.0, step=0.1)

        submitted = st.form_submit_button("Ajouter obligation")

        if submitted:
            # Validation
            erreurs    = []
            avertissements = []

            # Par
            if len(isin) < 3:
                avertissements.append("ISIN semble trop court")
            if prix < 50:
                avertissements.append("Prix < 50 — obligation en détresse ?")
            if prix > 200:
                avertissements.append("Prix > 200 — inhabituel")
            if date_maturite <= date_emission:
                erreurs.append("Date maturité doit être après date émission")
            if coupon > 20:
                avertissements.append("Coupon > 20% — vérifiez la valeur")

            for w in avertissements:
                st.warning(w)

            if erreurs:
                for e in erreurs:
                    st.error(e)
            else:
                # Ajouter l'obligation
                st.session_state['obligations_manuelles'].append({
                    'isin'            : isin,
                    'type_obligation' : type_obligation,
                    'Pays'            : pays,
                    'Coupon'          : coupon / 100,  # convertir en décimal
                    'prix'            : prix,
                    'devise'          : devise,
                    'date_emission'   : str(date_emission),
                    'date_maturite'   : str(date_maturite),
                    'ytm'             : ytm / 100,
                    'benchmark'       : None,
                })
                st.success(f"Obligation {isin} ajoutée !")

    # ── Tableau des obligations saisies ───────────────────────────────────────
    if st.session_state['obligations_manuelles']:
        st.markdown("### Obligations saisies")
        df_manuel = pd.DataFrame(st.session_state['obligations_manuelles'])
        st.dataframe(df_manuel, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Vider la liste"):
                st.session_state['obligations_manuelles'] = []
                st.rerun()

        with col2:
            if st.button("Lancer l'analyse", type="primary"):
                st.session_state['dataset']    = df_manuel
                st.session_state['data_prete'] = True
                st.success("Données prêtes ! Naviguez vers les pages d'analyse.")
                st.balloons()
    else:
        st.info("Aucune obligation saisie pour l'instant.")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — STATUT + ADMIN
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("---")
st.sidebar.markdown("### Statut")

if st.session_state['data_prete']:
    n = len(st.session_state['dataset'])
    st.sidebar.success(f"✓ {n} obligations chargées")
else:
    st.sidebar.warning("Aucune donnée chargée")

# ── Admin caché ───────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
password = st.sidebar.text_input("Accès Admin", type="password")

if password == ADMIN_PASSWORD:
    st.sidebar.success("✓ Accès administrateur")
    st.sidebar.markdown("### Paramètres Admin")

    st.sidebar.markdown("**Anticipations BC par défaut :**")
    for pays, bc in BANQUES_CENTRALES.items():
        val = st.sidebar.slider(
            f"{bc} ({pays}) %",
            min_value = -2.0,
            max_value =  2.0,
            value     = float(ANTICIPATIONS_BC_DEFAUT[pays] * 100),
            step      = 0.25,
            key       = f"admin_{pays}"
        )
        st.session_state['chocs_par_pays'][pays] = val / 100

    st.sidebar.markdown("**Paramètres ML :**")
    st.sidebar.write(f"Algorithme : AgglomerativeClustering")
    st.sidebar.write(f"Linkage : ward | Metric : euclidean")
    st.sidebar.write(f"k = 3 clusters")

    st.sidebar.markdown("**Paramètres ARIMA :**")
    st.sidebar.write(f"Ordre : (0, 1, 1)")
    st.sidebar.write(f"Ticker : ^TNX")
    st.sidebar.write(f"Période : 2015-2025")
