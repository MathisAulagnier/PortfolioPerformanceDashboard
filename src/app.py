# app.py
import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

# Import des modules refactorisés
import ui_components
import data_manager
import calculations
import ai_analyzer

# --- Configuration de la page ---
st.set_page_config(
    layout="wide",
    page_title="Dashboard de Backtesting"
)

# --- Initialisation de la session ---
if 'tickers_list' not in st.session_state:
    st.session_state.tickers_list = []
if 'weights' not in st.session_state:
    st.session_state.weights = {}
if 'portfolio_loaded' not in st.session_state:
    st.session_state.portfolio_loaded = False
if 'analysis_type' not in st.session_state:
    st.session_state.analysis_type = 'drawdown'


# --- Barre Latérale (gérée par un module dédié) ---
# La fonction render_sidebar retourne un dictionnaire avec tous les paramètres
config = ui_components.render_sidebar()

# --- Page Principale ---
st.title("Dashboard de Backtesting de Portefeuille")

# --- Vérifications initiales ---
if not st.session_state.tickers_list:
    st.info("Bienvenue ! Veuillez commencer par ajouter au moins une action dans la barre latérale.")
    st.stop()

total_weight_check = sum(st.session_state.weights.values())
if total_weight_check != 100:
    st.error(f"**Répartition invalide !** Le total des poids doit être de 100%, mais il est de {total_weight_check}%. Veuillez ajuster les poids dans la barre latérale.")
    st.stop()

# --- Récupération des données ---
all_tickers_to_fetch = st.session_state.tickers_list + [config['benchmark']]
all_data = data_manager.get_data(all_tickers_to_fetch, config['start_date'], config['end_date'])

if all_data.empty or all_data[st.session_state.tickers_list].isnull().all().all():
    st.warning("Aucune donnée disponible pour les actions sélectionnées sur la période choisie. Essayez une période plus courte ou d'autres actions.")
    st.stop()

# --- Calculs de Performance ---
valid_tickers = [t for t in st.session_state.tickers_list if t in all_data.columns and not all_data[t].isnull().all()]
portfolio_data = all_data[valid_tickers].dropna()

if portfolio_data.empty:
    st.warning("Les données du portefeuille sont vides après nettoyage. Impossible de continuer.")
    st.stop()

returns = portfolio_data.pct_change().dropna()
weights = [st.session_state.weights[ticker] / 100.0 for ticker in returns.columns]

portfolio_value, total_invested, portfolio_returns = calculations.calculate_dca_portfolio(
    returns, weights, config['initial_capital'], config['dca_enabled'], config['dca_amount'], config['dca_frequency']
)

benchmark_returns = all_data[config['benchmark']].reindex(portfolio_value.index).pct_change().dropna()

if config['dca_enabled']:
    benchmark_value, benchmark_total_invested, _ = calculations.calculate_dca_portfolio(
        benchmark_returns.to_frame(config['benchmark']), [1.0], config['initial_capital'], config['dca_enabled'], config['dca_amount'], config['dca_frequency']
    )
else:
    benchmark_value = config['initial_capital'] * (1 + benchmark_returns).cumprod()
    benchmark_total_invested = pd.Series([config['initial_capital']] * len(benchmark_returns), index=benchmark_returns.index)

# --- Création des onglets pour une meilleure navigation ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Performance & Métriques", 
    "📉 Risques & Horizon", 
    "📊 Composition du Portefeuille", 
    "🤖 Analyse par IA",
    "📚 Guides & Aide"
])

# --- Onglet 1: Performance & Métriques ---
with tab1:
    ui_components.render_performance_section(
        config, portfolio_value, total_invested, benchmark_value, portfolio_returns, benchmark_returns,
        benchmark_total_invested
    )

# --- Onglet 2: Risques & Horizon ---
with tab2:
    ui_components.render_risk_analysis_section(
        portfolio_value, benchmark_value, portfolio_returns
    )

# --- Onglet 3: Composition du Portefeuille ---
with tab3:
    ui_components.render_composition_section(valid_tickers)

# --- Onglet 4: Analyse par IA ---
with tab4:
    ui_components.render_ai_analysis_section(
        config, portfolio_value, total_invested, benchmark_value, portfolio_returns, 
        benchmark_returns, benchmark_total_invested, valid_tickers
    )

# --- Onglet 5: Guides & Aide ---
with tab5:
    ui_components.render_guides_section()