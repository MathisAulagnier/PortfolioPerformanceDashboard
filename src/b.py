import streamlit as st

import yfinance as yf

import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px

from datetime import datetime
from dateutil.relativedelta import relativedelta

import json


# --- Configuration de la page ---
st.set_page_config(
    layout="wide",
    page_title="Dashboard de Backtesting"
)

# --- Fonctions de calcul améliorées ---

@st.cache_data
def get_data(tickers, start, end):
    """Télécharge les données 'Close' pour une liste de tickers et les met en cache."""
    try:
        data = yf.download(tickers, start=start, end=end, progress=False)['Close']
        # Si un seul ticker est demandé, yf.download renvoie une Series, on la convertit en DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        return data.dropna(how='all')
    except Exception as e:
        st.error(f"Erreur lors du téléchargement des données : {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)  # Cache pendant 1 heure
def get_risk_free_rate():
    """
    Récupère le taux sans risque actuel (bons du Trésor américain 13 semaines).
    
    Returns:
        float: Taux sans risque annuel en décimal (ex: 0.05 pour 5%)
    """
    try:
        treasury_data = yf.download("^IRX", period="5d", progress=False)
        if not treasury_data.empty and 'Close' in treasury_data.columns:
            latest_rate = treasury_data['Close'].iloc[-1] / 100.0
            return float(latest_rate)
        else:
            return 0.0
    except Exception:
        return 0.0

def calculate_dca_portfolio(returns, weights, initial_capital, dca_enabled, dca_amount, dca_frequency):
    """Calcule l'évolution du portefeuille avec ou sans DCA."""
    if returns.empty:
        return pd.Series(), pd.Series(), pd.Series()
    portfolio_returns = (returns * weights).sum(axis=1)
    portfolio_value = pd.Series(index=returns.index, dtype=float)
    total_invested = pd.Series(index=returns.index, dtype=float)
    current_value = initial_capital
    cumulative_invested = initial_capital
    portfolio_value.iloc[0] = current_value
    total_invested.iloc[0] = cumulative_invested
    if dca_enabled:
        dca_dates = set()
        current_date = returns.index[0]
        end_date = returns.index[-1]
        while current_date <= end_date:
            available_dates = returns.index[returns.index >= current_date]
            if len(available_dates) > 0:
                dca_dates.add(available_dates[0])
            current_date += relativedelta(months=1) if dca_frequency == 'Mensuelle' else relativedelta(years=1)
    for i in range(1, len(returns)):
        date = returns.index[i]
        daily_return = portfolio_returns.iloc[i]
        if dca_enabled and date in dca_dates:
            current_value += dca_amount
            cumulative_invested += dca_amount
        current_value *= (1 + daily_return)
        portfolio_value.iloc[i] = current_value
        total_invested.iloc[i] = cumulative_invested
    return portfolio_value, total_invested, portfolio_returns

def calculate_metrics_with_dca(portfolio_returns, portfolio_value, total_invested, risk_free_rate=0.0):
    if portfolio_returns.empty or portfolio_returns.isnull().all():
        return 0, 0, 0, 0, 0
    simple_return = (portfolio_value.iloc[-1] / total_invested.iloc[-1]) - 1
    twr = (1 + portfolio_returns).prod() - 1
    num_days = len(portfolio_returns)
    if num_days == 0:
        return 0, 0, 0, 0, 0
    annualized_return = ((1 + twr) ** (252 / num_days)) - 1
    volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / (volatility + 1e-10)
    return simple_return, annualized_return, volatility, sharpe_ratio, twr

def calculate_advanced_metrics(portfolio_returns, benchmark_returns, risk_free_rate=0.0):
    if portfolio_returns.empty or benchmark_returns.empty:
        return {'sortino_ratio': 0, 'alpha': 0, 'beta': 0}
    aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner').dropna()
    if aligned_data.empty or len(aligned_data.columns) < 2:
        return {'sortino_ratio': 0, 'alpha': 0, 'beta': 0}
    portfolio_aligned = aligned_data.iloc[:, 0]
    benchmark_aligned = aligned_data.iloc[:, 1]
    daily_risk_free = risk_free_rate / 252
    negative_returns = portfolio_aligned[portfolio_aligned < daily_risk_free] - daily_risk_free
    downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 1e-10
    portfolio_annual_return = ((1 + portfolio_aligned.mean()) ** 252) - 1
    sortino_ratio = (portfolio_annual_return - risk_free_rate) / downside_deviation
    if benchmark_aligned.var() > 1e-10:
        covariance = np.cov(portfolio_aligned, benchmark_aligned)[0, 1]
        benchmark_variance = benchmark_aligned.var()
        beta = covariance / benchmark_variance
    else:
        beta = 0
    benchmark_annual_return = ((1 + benchmark_aligned.mean()) ** 252) - 1
    alpha = portfolio_annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
    return {'sortino_ratio': sortino_ratio, 'alpha': alpha, 'beta': beta}

def calculate_drawdown_series(portfolio_value):
    if portfolio_value.empty or portfolio_value.isnull().all():
        return pd.Series()
    peak = portfolio_value.expanding(min_periods=1).max()
    return (portfolio_value / peak) - 1

def calculate_max_drawdown(portfolio_value):
    if portfolio_value.empty or portfolio_value.isnull().all():
        return 0
    return calculate_drawdown_series(portfolio_value).min()

def calculate_holding_period_analysis(portfolio_returns, max_horizon_years=20):
    if portfolio_returns.empty or len(portfolio_returns) < 252:
        return {}
    results = {}
    cumulative_returns = (1 + portfolio_returns).cumprod()
    horizons = []
    for months in [6, 12, 18, 24, 30, 36]:
        if months <= max_horizon_years * 12:
            horizons.append((months / 12, f"{months}m"))
    for years in range(4, max_horizon_years + 1):
        horizons.append((years, f"{years}a"))
    for horizon_years, label in horizons:
        horizon_days = int(horizon_years * 252)
        if horizon_days >= len(cumulative_returns):
            continue
        period_returns = []
        for i in range(len(cumulative_returns) - horizon_days):
            start_value = cumulative_returns.iloc[i]
            end_value = cumulative_returns.iloc[i + horizon_days]
            period_returns.append((end_value / start_value) - 1)
        if not period_returns:
            continue
        period_returns = np.array(period_returns)
        positive_returns = period_returns[period_returns > 0]
        probability_positive = len(positive_returns) / len(period_returns) * 100
        avg_return = np.mean(period_returns)
        avg_positive_return = np.mean(positive_returns) if len(positive_returns) > 0 else 0
        avg_negative_return = np.mean(period_returns[period_returns <= 0]) if np.any(period_returns <= 0) else 0
        percentile_10 = np.percentile(period_returns, 10)
        percentile_90 = np.percentile(period_returns, 90)
        median_return = np.median(period_returns)
        results[horizon_years] = {
            'label': label,
            'probability_positive': probability_positive,
            'avg_return': avg_return,
            'median_return': median_return,
            'avg_positive_return': avg_positive_return,
            'avg_negative_return': avg_negative_return,
            'percentile_10': percentile_10,
            'percentile_90': percentile_90,
            'num_periods': len(period_returns),
            'annualized_avg_return': ((1 + avg_return) ** (1/horizon_years)) - 1 if horizon_years > 0 else avg_return
        }
    return results

# --- Barre Latérale (Sidebar) pour tous les contrôles ---

st.sidebar.title("Paramètres du Backtest")
st.sidebar.markdown("Commencez par ajouter des actions à votre portefeuille.")

# Initialisation de la mémoire de l'application
if 'tickers_list' not in st.session_state:
    st.session_state.tickers_list = []
if 'weights' not in st.session_state:
    st.session_state.weights = {}
if 'portfolio_loaded' not in st.session_state:
    st.session_state.portfolio_loaded = False

# Section pour ajouter un ticker
with st.sidebar.form("ticker_form", clear_on_submit=True):
    new_ticker_input = st.text_input("Entrez un ticker (ex: 'AAPL', 'KER.PA')", key="ticker_input")
    submitted = st.form_submit_button("Ajouter et Valider")

    if submitted and new_ticker_input:
        ticker_to_add = new_ticker_input.strip().upper()
        if ticker_to_add not in st.session_state.tickers_list:
            test_data = yf.Ticker(ticker_to_add).history(period="1mo")
            if test_data.empty:
                st.sidebar.error(f"Ticker '{ticker_to_add}' invalide ou sans données.")
            else:
                st.sidebar.success(f"'{ticker_to_add}' ajouté au portefeuille !")
                st.session_state.tickers_list.append(ticker_to_add)
                st.session_state.weights[ticker_to_add] = 0
        else:
            st.sidebar.warning(f"'{ticker_to_add}' est déjà dans la liste.")

# Section pour afficher et gérer le portefeuille actuel
if st.session_state.tickers_list:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Portefeuille Actuel")
    
    # Nettoyage des poids pour les tickers qui n'existent plus
    st.session_state.weights = {ticker: st.session_state.weights.get(ticker, 0) for ticker in st.session_state.tickers_list}

    for ticker in st.session_state.tickers_list:
        st.sidebar.markdown(f"**{ticker}**")
    
    # --- NOUVEAU : Section Sauvegarde/Chargement ---
    st.sidebar.markdown("**Gestion du Portefeuille**")
    
    # Sauvegarde du portefeuille
    portfolio_data = {
        "tickers": st.session_state.tickers_list,
        "weights": st.session_state.weights,
        "saved_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": "1.0"
    }
    
    portfolio_json = json.dumps(portfolio_data, indent=2, ensure_ascii=False)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.download_button(
            label="Sauvegarder",
            data=portfolio_json,
            file_name=f"portefeuille_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            help="Télécharge la configuration actuelle du portefeuille"
        )
    
    with col2:
        if st.button("Vider", type="secondary", help="Vide complètement le portefeuille"):
            st.session_state.tickers_list = []
            st.session_state.weights = {}
            st.rerun()

# --- NOUVEAU : Section Chargement de Portefeuille ---
st.sidebar.markdown("---")
st.sidebar.subheader("Charger un Portefeuille")

# Initialiser la variable de contrôle pour éviter les rechargements multiples
if 'portfolio_loaded' not in st.session_state:
    st.session_state.portfolio_loaded = False

uploaded_file = st.sidebar.file_uploader(
    "Choisissez un fichier de portefeuille",
    type=['json'],
    help="Chargez un fichier JSON de portefeuille précédemment sauvegardé",
    key="portfolio_uploader"
)

if uploaded_file is not None and not st.session_state.portfolio_loaded:
    try:
        # Lecture et décodage du fichier JSON
        portfolio_data = json.load(uploaded_file)
        
        # Validation des données
        if "tickers" in portfolio_data and "weights" in portfolio_data:
            # Validation que les tickers existent toujours
            valid_tickers = []
            valid_weights = {}
            
            for ticker in portfolio_data["tickers"]:
                try:
                    # Test rapide pour vérifier que le ticker existe encore
                    test_data = yf.Ticker(ticker).history(period="5d")
                    if not test_data.empty:
                        valid_tickers.append(ticker)
                        valid_weights[ticker] = portfolio_data["weights"].get(ticker, 0)
                    else:
                        st.sidebar.warning(f"Ticker '{ticker}' n'est plus valide et a été ignoré.")
                except:
                    st.sidebar.warning(f"Impossible de valider le ticker '{ticker}', ignoré.")
            
            if valid_tickers:
                # Mise à jour du state
                st.session_state.tickers_list = valid_tickers
                st.session_state.weights = valid_weights
                st.session_state.portfolio_loaded = True
                
                # Information sur le chargement
                loaded_date = portfolio_data.get("saved_date", "Inconnue")
                st.sidebar.success(f"Portefeuille chargé avec succès !")
                st.sidebar.info(f"Sauvegardé le: {loaded_date}")
                st.sidebar.info(f"Tickers chargés: {len(valid_tickers)}")
                
                # Forcer le rechargement de la page pour mettre à jour l'affichage
                st.rerun()
            else:
                st.sidebar.error("Aucun ticker valide trouvé dans le fichier.")
        else:
            st.sidebar.error("Format de fichier invalide. Le fichier doit contenir 'tickers' et 'weights'.")
    
    except json.JSONDecodeError:
        st.sidebar.error("Erreur de lecture du fichier JSON. Vérifiez le format du fichier.")
    except Exception as e:
        st.sidebar.error(f"Erreur lors du chargement du portefeuille: {str(e)}")

# Réinitialiser le flag quand il n'y a plus de fichier uploadé
elif uploaded_file is None and st.session_state.portfolio_loaded:
    st.session_state.portfolio_loaded = False

# Section pour définir la répartition
if st.session_state.tickers_list:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Répartition du Portefeuille (%)")
    
    total_weight = 0
    for ticker in st.session_state.tickers_list:
        weight = st.sidebar.number_input(
            f"Poids pour {ticker}",
            min_value=0,
            max_value=100,
            value=st.session_state.weights.get(ticker, 0),
            step=5,
            key=f"weight_{ticker}"
        )
        st.session_state.weights[ticker] = weight
        total_weight += weight

    st.sidebar.markdown(f"**Total alloué : {total_weight}%**")
    if total_weight != 100:
        st.sidebar.warning("Le total des répartitions doit être égal à 100% pour lancer le backtest.")

st.sidebar.markdown("---")

# --- NOUVELLE SECTION : Configuration DCA ---
st.sidebar.subheader("Investissement Programmé (DCA)")
dca_enabled = st.sidebar.checkbox("Activer l'investissement programmé", value=False)

if dca_enabled:
    dca_frequency = st.sidebar.selectbox(
        "Fréquence d'investissement",
        ["Mensuelle", "Annuelle"],
        index=0
    )
    
    dca_amount = st.sidebar.number_input(
        "Montant à ajouter ($)",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        help="Montant qui sera ajouté automatiquement selon la fréquence choisie"
    )
    
    st.sidebar.info(f"Avec ce paramétrage, {dca_amount}$ seront ajoutés au portefeuille chaque période ({dca_frequency.lower()}).")
else:
    dca_frequency = "Mensuelle"  # Valeur par défaut
    dca_amount = 0

st.sidebar.markdown("---")

# Section pour la configuration du backtest (capital, benchmark)
st.sidebar.subheader("Configuration Générale")
initial_capital = st.sidebar.number_input("Capital Initial ($)", min_value=1000, value=10000, step=1000)
benchmark = st.sidebar.selectbox(
    "Indice de référence",
    ["^GSPC", "^IXIC", "GC=F", "DX-Y.NYB"],
    format_func=lambda x: {"^GSPC": "S&P 500", "^IXIC": "NASDAQ", "GC=F": "Or", "DX-Y.NYB": "Dollar Index"}[x]
)

# --- NOUVEAU : Configuration du taux sans risque ---
st.sidebar.markdown("---")
st.sidebar.subheader("Taux Sans Risque")

# Récupération automatique du taux sans risque
auto_risk_free_rate = get_risk_free_rate()

# Option pour utiliser le taux automatique ou manuel
use_auto_rate = st.sidebar.checkbox(
    "Utiliser le taux automatique (Bons du Trésor US 13 semaines)", 
    value=True,
    help="Si coché, utilise le taux des bons du Trésor américain récupéré automatiquement"
)

if use_auto_rate and auto_risk_free_rate > 0:
    risk_free_rate = auto_risk_free_rate
    st.sidebar.success(f"Taux automatique: {risk_free_rate:.2%}")
else:
    # Input manuel si le taux automatique échoue ou si l'option manuelle est choisie
    risk_free_rate = st.sidebar.number_input(
        "Taux sans risque annuel (%)",
        min_value=0.0,
        max_value=20.0,
        value=auto_risk_free_rate * 100 if auto_risk_free_rate > 0 else 0.0,
        step=0.1,
        help="Taux sans risque utilisé pour les calculs d'Alpha et de Sharpe"
    ) / 100.0  # Conversion en décimal
    
    if not use_auto_rate:
        st.sidebar.info(f"Taux manuel: {risk_free_rate:.2%}")
    elif auto_risk_free_rate == 0:
        st.sidebar.warning("Impossible de récupérer le taux automatique. Utilisez le taux manuel.")

# --- Page Principale pour l'affichage des résultats ---

# Vérification de la répartition du portefeuille
total_weight_check = sum(st.session_state.weights.values())
if total_weight_check != 100:
    st.error(f"**Répartition invalide !** Le total des poids doit être de 100%, mais il est de {total_weight_check}%. Veuillez ajuster les poids dans la barre latérale.")
    st.stop()

# --- Section de sélection de la période ---
st.markdown("### Choisissez une période de visualisation")
periods = {"1 an": 12, "3 ans": 36, "5 ans": 60, "10 ans": 120, "20 ans": 240, "Max": None}
selected_period = st.radio("Période", options=list(periods.keys()), horizontal=True, label_visibility="collapsed")

end_date = datetime.now()
if periods[selected_period] is not None:
    start_date = end_date - relativedelta(months=periods[selected_period])
else:
    start_date = datetime(1995, 1, 1)

all_tickers_to_fetch = st.session_state.tickers_list + [benchmark]
all_data = get_data(all_tickers_to_fetch, start_date, end_date)

if all_data.empty or all_data[st.session_state.tickers_list].isnull().all().all():
    st.warning("Aucune donnée disponible pour les actions sélectionnées sur la période choisie.")
    st.stop()

# --- Calculs de performance avec DCA ---
valid_tickers = [t for t in st.session_state.tickers_list if t in all_data.columns and not all_data[t].isnull().all()]
portfolio_data = all_data[valid_tickers].dropna()

if portfolio_data.empty:
    st.warning("Les données du portefeuille sont vides après nettoyage. Impossible de continuer.")
    st.stop()
    
returns = portfolio_data.pct_change().dropna()

# Utilisation des poids personnalisés
weights = [st.session_state.weights[ticker] / 100.0 for ticker in returns.columns]

# --- NOUVEAU : Calcul avec DCA ---
portfolio_value, total_invested, portfolio_returns = calculate_dca_portfolio(
    returns, weights, initial_capital, dca_enabled, dca_amount, dca_frequency
)

# --- CORRECTION : Calcul du benchmark avec DCA si activé ---
benchmark_returns = all_data[benchmark].reindex(portfolio_value.index).pct_change().dropna()

if dca_enabled:
    # Appliquer la même logique DCA au benchmark
    benchmark_value, benchmark_total_invested, _ = calculate_dca_portfolio(
        benchmark_returns.to_frame(benchmark), [1.0], initial_capital, dca_enabled, dca_amount, dca_frequency
    )
else:
    # Calcul standard sans DCA
    benchmark_value = initial_capital * (1 + benchmark_returns).cumprod()
    benchmark_total_invested = pd.Series([initial_capital] * len(benchmark_returns), index=benchmark_returns.index)

# Métriques principales
p_simple, p_annual, p_vol, p_sharpe, p_twr = calculate_metrics_with_dca(portfolio_returns, portfolio_value, total_invested, risk_free_rate)
p_drawdown = calculate_max_drawdown(portfolio_value)
if dca_enabled:
    benchmark_simple = (benchmark_value.iloc[-1] / benchmark_total_invested.iloc[-1]) - 1 if len(benchmark_value) > 0 else 0
    benchmark_twr = (1 + benchmark_returns).prod() - 1 if len(benchmark_returns) > 0 else 0
    num_days = len(benchmark_returns)
    b_annual = ((1 + benchmark_twr) ** (252 / num_days)) - 1 if num_days > 0 else 0
    b_total = benchmark_simple
else:
    b_total = (benchmark_value.iloc[-1] / initial_capital) - 1 if len(benchmark_value) > 0 else 0
    num_days = len(benchmark_returns)
    b_annual = ((1 + b_total) ** (252 / num_days)) - 1 if num_days > 0 else 0
b_vol = benchmark_returns.std() * np.sqrt(252) if len(benchmark_returns) > 0 else 0
b_sharpe = (b_annual - risk_free_rate) / (b_vol + 1e-10)
b_drawdown = calculate_max_drawdown(benchmark_value)
advanced_metrics = calculate_advanced_metrics(portfolio_returns, benchmark_returns, risk_free_rate)
portfolio_drawdown_series = calculate_drawdown_series(portfolio_value)
benchmark_drawdown_series = calculate_drawdown_series(benchmark_value)

# ================= FONCTIONS DE RENDU PAR ONGLET =================

def render_overview():
    st.subheader("Vue d'ensemble")
    if dca_enabled:
        total_dca_added = total_invested.iloc[-1] - initial_capital
        st.info(f"**DCA activé** : {total_dca_added:,.0f}$ ajoutés sur la période")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value, mode='lines', name='Portefeuille', line=dict(color='royalblue', width=2)))
    fig.add_trace(go.Scatter(x=benchmark_value.index, y=benchmark_value, mode='lines', name=f'Indice ({benchmark})', line=dict(color='grey', width=2, dash='dash')))
    if dca_enabled:
        fig.add_trace(go.Scatter(x=total_invested.index, y=total_invested, mode='lines', name='Capital Investi', line=dict(color='orange', width=1, dash='dot')))
    fig.update_layout(title=f"Évolution du capital ({selected_period})", xaxis_title="Date", yaxis_title="Valeur ($)", hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Rendement Total", f"{p_simple:.2%}")
    col2.metric("Rendement Annualisé", f"{p_annual:.2%}")
    col3.metric("Volatilité", f"{p_vol:.2%}")
    col4.metric("Sharpe", f"{p_sharpe:.2f}")
    col5.metric("Max Drawdown", f"{p_drawdown:.2%}")


def render_performance_risks():
    st.subheader("Performance & Risques")
    col_info1, col_info2 = st.columns([3,1])
    with col_info1:
        if use_auto_rate and auto_risk_free_rate > 0:
            st.info(f"Taux sans risque utilisé: {risk_free_rate:.2%} (auto)")
        else:
            st.info(f"Taux sans risque utilisé: {risk_free_rate:.2%} (manuel)")
    with col_info2:
        if st.button("Actualiser taux"):
            st.cache_data.clear(); st.rerun()
    # Toggle Drawdown / Horizon
    st.markdown("### Analyse des Risques et Horizon")
    colA, colB = st.columns(2)
    if 'analysis_type' not in st.session_state:
        st.session_state.analysis_type = 'drawdown'
    with colA:
        if st.button("Évolution du Drawdown", type="primary" if st.session_state.analysis_type=='drawdown' else "secondary", use_container_width=True):
            st.session_state.analysis_type='drawdown'
    with colB:
        if st.button("Horizon de Placement", type="primary" if st.session_state.analysis_type=='horizon' else "secondary", use_container_width=True):
            st.session_state.analysis_type='horizon'
    if st.session_state.analysis_type=='drawdown':
        dd_port = portfolio_drawdown_series; dd_bench = benchmark_drawdown_series
        if not dd_port.empty:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=dd_port.index, y=dd_port*100, mode='lines', name='Portefeuille', line=dict(color='red', width=2), fill='tonexty', fillcolor='rgba(255,0,0,0.1)'))
            if not dd_bench.empty:
                fig_dd.add_trace(go.Scatter(x=dd_bench.index, y=dd_bench*100, mode='lines', name=f'Benchmark {benchmark}', line=dict(color='grey', width=2, dash='dash'), fill='tonexty', fillcolor='rgba(128,128,128,0.05)'))
            fig_dd.add_hline(y=0, line_color='black', line_width=1)
            max_dd_date = dd_port.idxmin(); max_dd_val = dd_port.min()
            recovery_info = ""
            if max_dd_val < -0.01:
                peak_before = portfolio_value[:max_dd_date].max()
                rec_dates = portfolio_value[max_dd_date:][portfolio_value[max_dd_date:] >= peak_before]
                if not rec_dates.empty:
                    recovery_days = (rec_dates.index[0]-max_dd_date).days
                    recovery_info = f" | Récupération: {recovery_days} j"
            st.info(f"Drawdown max {max_dd_val:.2%} le {max_dd_date.strftime('%d/%m/%Y')}{recovery_info}")
            st.plotly_chart(fig_dd, use_container_width=True)
            c1,c2,c3,c4 = st.columns(4)
            avg_dd = dd_port[dd_port<0].mean() if (dd_port<0).any() else 0
            with c1: st.metric("DD Moyen", f"{avg_dd:.2%}")
            with c2: st.metric("Jours DD>5%", f"{(dd_port<-0.05).sum()}")
            # temps de recup moyenne
            dd_start = dd_port[(dd_port.shift(1)>=0)&(dd_port<0)].index
            rec_times=[]
            for sd in dd_start:
                peak_v = portfolio_value[sd]
                fut = portfolio_value[sd:]
                rec_idx = fut[fut>=peak_v]
                if not rec_idx.empty:
                    rec_times.append((rec_idx.index[0]-sd).days)
            avg_rec = np.mean(rec_times) if rec_times else 0
            with c3: st.metric("Récup. Moy", f"{avg_rec:.0f}j")
            with c4: st.metric("Temps en DD", f"{(dd_port<-0.01).sum()/len(dd_port)*100:.1f}%")
        else:
            st.warning("Données insuffisantes pour le drawdown.")
    else:
        # Horizon placement
        if selected_period == "Max":
            max_horizon = min(20, len(portfolio_returns)//252)
        elif selected_period == "20 ans": max_horizon=20
        elif selected_period == "10 ans": max_horizon=10
        elif selected_period == "5 ans": max_horizon=5
        else: max_horizon = min(10, len(portfolio_returns)//252)
        max_horizon = max(2, max_horizon)
        holding_analysis = calculate_holding_period_analysis(portfolio_returns, max_horizon)
        if holding_analysis:
            horizons = sorted(holding_analysis.keys())
            probabilities = [holding_analysis[h]['probability_positive'] for h in horizons]
            labels = [holding_analysis[h]['label'] for h in horizons]
            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(x=horizons,y=probabilities,mode='lines+markers',name='Probabilité Gain',line=dict(color='green',width=3),marker=dict(size=8)))
            fig_h.add_hline(y=50,line_dash='dash',line_color='orange')
            fig_h.add_hline(y=80,line_dash='dot',line_color='blue')
            fig_h.update_layout(title="Probabilité de Gain vs Horizon", xaxis_title="Années", yaxis_title="Probabilité (%)", yaxis=dict(range=[0,105]))
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.warning("Période insuffisante pour analyser l'horizon.")
    st.markdown("---")
    st.markdown("### Métriques Avancées")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.metric("Sortino", f"{advanced_metrics['sortino_ratio']:.2f}")
    with c2:
        st.metric("Bêta", f"{advanced_metrics['beta']:.2f}")
    with c3:
        st.metric("Alpha", f"{advanced_metrics['alpha']:.2%}")
    if not portfolio_returns.empty and not benchmark_returns.empty:
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner').dropna()
        if not aligned.empty and len(aligned.columns)>=2:
            x = aligned.iloc[:,1]*100; y = aligned.iloc[:,0]*100
            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(x=x,y=y,mode='markers',name='Journaliers',marker=dict(size=4,opacity=0.6,color='royalblue')))
            if len(x)>1:
                slope, intercept = np.polyfit(x,y,1)
                line_x = np.array([x.min(), x.max()]); line_y = slope*line_x+intercept
                fig_reg.add_trace(go.Scatter(x=line_x,y=line_y,mode='lines',name=f'Vertical (β={slope:.2f})',line=dict(color='red')))
            fig_reg.update_layout(title=f"Corrélation vs {benchmark}", xaxis_title=f"{benchmark} (%)", yaxis_title="Portefeuille (%)")
            st.plotly_chart(fig_reg, use_container_width=True)


def render_allocation_diversification():
    st.subheader("Allocation & Diversification")
    # Géographie
    geo_data=[]
    for t in valid_tickers:
        try:
            info = yf.Ticker(t).info; country = info.get('country','Inconnu')
            geo_data.append({'Ticker':t,'Country':country,'Weight':st.session_state.weights.get(t,0)})
        except Exception: pass
    geo_df = pd.DataFrame(geo_data)
    if not geo_df.empty:
        geo_df = geo_df.groupby('Country').agg({'Weight':'sum'}).reset_index(); geo_df=geo_df[geo_df['Weight']>0]
    if not geo_df.empty:
        gc1,gc2 = st.columns(2)
        with gc1:
            fig_geo = px.pie(geo_df, names='Country', values='Weight', title='Répartition par Pays')
            fig_geo.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_geo, use_container_width=True)
        with gc2:
            conflict_countries = ['Ukraine','Russia','Afghanistan','Syria','Yemen','Myanmar','Ethiopia','Sudan','Gaza','Israel','Palestine']
            fig_map = px.choropleth(geo_df, locations='Country', locationmode='country names', hover_name='Country', title='Exposition Géographique', color='Weight', color_continuous_scale=px.colors.sequential.Plasma)
            fig_map.add_trace(go.Choropleth(locations=conflict_countries, locationmode='country names', z=[1]*len(conflict_countries), colorscale=[[0,'black'],[1,'black']], showscale=False, name='Zones de conflit'))
            fig_map.update_geos(fitbounds="locations", visible=True)
            st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("Pas de données géographiques disponibles.")
    st.markdown("---")
    # Secteurs & Industries
    sector_data=[]; industry_data=[]
    for t in valid_tickers:
        try:
            info=yf.Ticker(t).info
            sector_data.append({'Ticker':t,'Sector':info.get('sectorKey','Inconnu'),'Weight':st.session_state.weights.get(t,0)})
            industry_data.append({'Ticker':t,'Industry':info.get('industryKey','Inconnu'),'Weight':st.session_state.weights.get(t,0)})
        except Exception:
            sector_data.append({'Ticker':t,'Sector':'Inconnu','Weight':st.session_state.weights.get(t,0)})
            industry_data.append({'Ticker':t,'Industry':'Inconnu','Weight':st.session_state.weights.get(t,0)})
    if sector_data:
        sector_df=pd.DataFrame(sector_data)
        sector_summary=sector_df.groupby('Sector').agg({'Weight':'sum','Ticker':lambda x:', '.join(x)}).reset_index(); sector_summary=sector_summary[sector_summary['Weight']>0]
    else:
        sector_summary=pd.DataFrame()
    if industry_data:
        industry_df=pd.DataFrame(industry_data)
        industry_summary=industry_df.groupby('Industry').agg({'Weight':'sum','Ticker':lambda x:', '.join(x)}).reset_index(); industry_summary=industry_summary[industry_summary['Weight']>0]
    else:
        industry_summary=pd.DataFrame()
    if not sector_summary.empty and not industry_summary.empty:
        sc1, sc2 = st.columns(2)
        with sc1:
            fig_sector = px.pie(sector_summary, names='Sector', values='Weight', title='Secteurs', hover_data=['Ticker'])
            fig_sector.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_sector, use_container_width=True)
        with sc2:
            fig_industry = px.pie(industry_summary, names='Industry', values='Weight', title='Industries', hover_data=['Ticker'])
            fig_industry.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_industry, use_container_width=True)
    if not sector_summary.empty:
        st.markdown("#### Détails Secteurs")
        display = sector_summary.rename(columns={'Sector':'Secteur','Weight':'Poids (%)','Ticker':'Tickers'}).copy()
        display['Poids (%)'] = display['Poids (%)'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(display, use_container_width=True, hide_index=True)
        num_sectors=len(sector_summary); max_sector_weight=sector_summary['Weight'].max(); dominant_sector=sector_summary.loc[sector_summary['Weight'].idxmax(),'Sector']
        hhi=sum((w/100)**2 for w in sector_summary['Weight'])
        mc1,mc2,mc3,mc4=st.columns(4)
        mc1.metric("Nb Secteurs", num_sectors)
        mc2.metric("Secteur Dominant", dominant_sector)
        mc3.metric("Poids Max", f"{max_sector_weight:.1f}%")
        mc4.metric("Indice HHI", f"{hhi:.3f}")


def render_strategy_dca():
    st.subheader("Stratégie & DCA")
    if dca_enabled:
        col1,col2,col3 = st.columns(3)
        with col1: st.metric("Capital Initial", f"{initial_capital:,.0f}$")
        with col2: st.metric("Total Ajouté (DCA)", f"{(total_invested.iloc[-1]-initial_capital):,.0f}$")
        with col3: st.metric("Capital Total Investi", f"{total_invested.iloc[-1]:,.0f}$")
        st.markdown("### Explication")
        st.write("Le DCA (Dollar-Cost Averaging) lisse vos points d'entrée en investissant périodiquement un montant fixe.")
    else:
        st.info("Le DCA n'est pas activé dans la configuration actuelle.")

# --- Monte Carlo Simulation Function conservée ---
@st.cache_data(show_spinner=False)
def run_monte_carlo_simulation(daily_portfolio_returns: pd.Series, horizon_years: int, initial_capital: float, num_simulations: int, manual_annual_return: float | None = None, manual_annual_vol: float | None = None, random_seed: int | None = 42) -> pd.DataFrame:
    if daily_portfolio_returns is None or daily_portfolio_returns.empty:
        return pd.DataFrame()
    np.random.seed(random_seed)
    trading_days = 252
    n_days = horizon_years * trading_days
    dt = 1 / trading_days
    hist_daily_mean = daily_portfolio_returns.mean()
    hist_daily_vol = daily_portfolio_returns.std()
    hist_annual_return = (1 + hist_daily_mean) ** trading_days - 1
    hist_annual_vol = hist_daily_vol * np.sqrt(trading_days)
    if manual_annual_return is not None and manual_annual_vol is not None:
        annual_return = manual_annual_return; annual_vol = manual_annual_vol
    else:
        annual_return = float(hist_annual_return); annual_vol = float(hist_annual_vol)
    mu_daily = (1 + annual_return) ** (1 / trading_days) - 1
    sigma_daily = annual_vol / np.sqrt(trading_days)
    drift = (mu_daily - 0.5 * sigma_daily ** 2) * dt
    diffusion_scale = sigma_daily * np.sqrt(dt)
    shocks = np.random.normal(0.0, 1.0, size=(n_days, num_simulations))
    log_returns = drift + diffusion_scale * shocks
    cumulative_log = np.cumsum(log_returns, axis=0)
    price_paths = initial_capital * np.exp(cumulative_log)
    index = pd.RangeIndex(start=1, stop=n_days + 1, step=1, name="Jour")
    return pd.DataFrame(price_paths, index=index)

def render_monte_carlo():
    st.subheader("🔮 Prédictions & Simulations (Monte-Carlo)")
    if portfolio_returns.empty:
        st.warning("Rendements historiques insuffisants pour la simulation.")
        return
    col_cfg1,col_cfg2,col_cfg3 = st.columns(3)
    with col_cfg1:
        horizon_years = st.slider("Horizon (années)",1,20,10,1)
    with col_cfg2:
        num_simulations = st.select_slider("Simulations", options=[200,500,1000,2000,3000,5000], value=1000)
    with col_cfg3:
        st.metric("Capital Initial", f"{initial_capital:,.0f}$")
    assumption_mode = st.radio("Hypothèses", ["Basées sur l'historique","Manuelles"], horizontal=True)
    manual_return=None; manual_vol=None
    if assumption_mode == "Manuelles":
        m1,m2 = st.columns(2)
        with m1:
            manual_return = st.number_input("Rendement ann. attendu (%)", -50.0, 100.0, 8.0, 0.5)/100
        with m2:
            manual_vol = st.number_input("Volatilité ann. (%)", 1.0, 100.0, 20.0, 0.5)/100
    if st.button("Lancer la simulation", type="primary"):
        with st.spinner("Simulation en cours..."):
            simulations_df = run_monte_carlo_simulation(portfolio_returns.squeeze(), horizon_years, initial_capital, num_simulations, manual_return, manual_vol)
        if simulations_df.empty:
            st.error("Simulation impossible.")
            return
        # Affichage
        max_display = min(100, simulations_df.shape[1])
        sample_cols = np.random.choice(simulations_df.columns, size=max_display, replace=False)
        sample_df = simulations_df[sample_cols]
        percentiles = simulations_df.quantile([0.05,0.5,0.95], axis=1).T
        percentiles.columns = ['p5','p50','p95']
        fig_mc = go.Figure()
        for c in sample_df.columns:
            fig_mc.add_trace(go.Scatter(x=sample_df.index, y=sample_df[c], mode='lines', line=dict(color='rgba(100,100,200,0.15)', width=1), hoverinfo='skip', showlegend=False))
        fig_mc.add_trace(go.Scatter(x=percentiles.index, y=percentiles['p5'], mode='lines', name='Pessimiste (5%)', line=dict(color='red',width=2)))
        fig_mc.add_trace(go.Scatter(x=percentiles.index, y=percentiles['p50'], mode='lines', name='Médiane (50%)', line=dict(color='blue',width=3)))
        fig_mc.add_trace(go.Scatter(x=percentiles.index, y=percentiles['p95'], mode='lines', name='Optimiste (95%)', line=dict(color='green',width=2)))
        fig_mc.update_layout(title=f"Distribution des Valeurs Futures ({horizon_years} ans)", xaxis_title='Jours', yaxis_title='Valeur ($)', hovermode='x unified')
        st.plotly_chart(fig_mc, use_container_width=True)
        final_values = simulations_df.iloc[-1]
        p5_val = final_values.quantile(0.05); p50_val = final_values.quantile(0.50); p95_val = final_values.quantile(0.95); mean_val = final_values.mean()
        cagr_median = (p50_val/initial_capital)**(1/horizon_years)-1; cagr_mean = (mean_val/initial_capital)**(1/horizon_years)-1
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Pessimiste (5%)", f"{p5_val:,.0f}$")
        m2.metric("Médiane", f"{p50_val:,.0f}$", f"CAGR {cagr_median:.2%}")
        m3.metric("Optimiste (95%)", f"{p95_val:,.0f}$")
        m4.metric("Moyenne", f"{mean_val:,.0f}$", f"CAGR {cagr_mean:.2%}")
        st.markdown("### Probabilité d'Atteindre un Objectif")
        target_value = st.number_input("Objectif ($)", min_value=0, value=int(initial_capital*2), step=1000)
        if target_value>0:
            prob_target = (final_values>=target_value).mean()
            st.info(f"Probabilité d'atteindre {target_value:,.0f}$ : **{prob_target:.1%}**")
        with st.expander("Histogramme des valeurs finales"):
            hist_fig = px.histogram(final_values, nbins=60, title="Distribution des valeurs finales", labels={'value':'Valeur finale ($)'})
            hist_fig.add_vline(x=mean_val, line_dash='dash', line_color='blue')
            hist_fig.add_vline(x=p50_val, line_dash='dot', line_color='black')
            hist_fig.add_vrect(x0=p5_val, x1=p95_val, fillcolor='green', opacity=0.08, line_width=0)
            st.plotly_chart(hist_fig, use_container_width=True)
        with st.expander("Méthodologie & Interprétation"):
            st.markdown("""Simulation Monte-Carlo via GBM (rendements log-normaux, volatilité constante). Limites: pas de régimes de marché, pas de queues grasses, volatilité non conditionnelle.""")

# --- IA ---

def generate_portfolio_analysis(portfolio_data, benchmark_data, metrics, advanced_metrics, period, tickers_list, weights, additional_context=None):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        assets_details={}; total_market_cap=0
        for ticker in tickers_list:
            try:
                info=yf.Ticker(ticker).info; mc=info.get('marketCap',0); total_market_cap+=mc
                assets_details[ticker]={"poids":float(weights.get(ticker,0)),"secteur":str(info.get('sectorKey','Inconnu')),"industrie":str(info.get('industryKey','Inconnu')),"pays":str(info.get('country','Inconnu')),"capitalisation":int(mc) if mc else 0}
            except:
                assets_details[ticker]={"poids":float(weights.get(ticker,0)),"secteur":"Inconnu","industrie":"Inconnu","pays":"Inconnu","capitalisation":0}
        secteurs={}; pays={}
        for t,d in assets_details.items():
            secteurs[d['secteur']]=secteurs.get(d['secteur'],0)+d['poids']
            pays[d['pays']]=pays.get(d['pays'],0)+d['poids']
        drawdown_series = calculate_drawdown_series(portfolio_data)
        drawdown_stats={}
        if not drawdown_series.empty:
            dd_moy = drawdown_series[drawdown_series<0].mean() if (drawdown_series<0).any() else 0
            drawdown_stats={"drawdown_moyen":f"{dd_moy:.2%}"}
        analysis_data={"periode":str(period),"actifs":assets_details,"metriques":{"ret_total":f"{metrics['portfolio_simple']:.2%}","ret_ann":f"{metrics['portfolio_annual']:.2%}","vol":f"{metrics['portfolio_vol']:.2%}","sharpe":f"{metrics['portfolio_sharpe']:.2f}","alpha":f"{advanced_metrics['alpha']:.2%}","beta":f"{advanced_metrics['beta']:.2f}"},"diversification":{"secteurs":secteurs,"pays":pays},"drawdown":drawdown_stats}
        prompt=f"Analyse experte du portefeuille suivant en français:\n{json.dumps(analysis_data, ensure_ascii=False, indent=2)}\nFournir: Performance, Risques, Diversification, Recommandations concrètes."
        response=client.chat.completions.create(model="gpt-3.5-turbo",messages=[{"role":"system","content":"Expert financier"},{"role":"user","content":prompt}],max_tokens=1200,temperature=0.7)
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur IA: {e}"

def render_ai_analysis():
    st.subheader("Analyse IA")
    if st.button("Générer l'analyse IA", type="primary"):
        with st.spinner("Analyse en cours..."):
            metrics_for_ai={ 'portfolio_simple':p_simple,'portfolio_annual':p_annual,'portfolio_vol':p_vol,'portfolio_sharpe':p_sharpe,'portfolio_drawdown':p_drawdown,'benchmark_total':b_total,'benchmark_annual':b_annual,'benchmark_vol':b_vol,'benchmark_sharpe':b_sharpe,'benchmark_drawdown':b_drawdown }
            ai_report=generate_portfolio_analysis(portfolio_value, benchmark_value, metrics_for_ai, advanced_metrics, selected_period, st.session_state.tickers_list, st.session_state.weights, additional_context={"risk_free_rate":risk_free_rate})
        st.markdown(ai_report)
        st.download_button("Télécharger", ai_report, file_name=f"analyse_portefeuille_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with st.expander("À propos"):
        st.write("Analyse générée via modèle OpenAI. Coût variable selon longueur du prompt.")

# ===================== STRUCTURE DES ONGLETES PRINCIPAUX =====================
main_tabs = st.tabs([
    "🏠 Vue d'ensemble",
    "📈 Performance & Risques",
    "🧩 Allocation & Diversification",
    "⏳ Stratégie & DCA",
    "🔮 Prédictions & Simulations",
    "🧠 Analyse IA"
])
with main_tabs[0]:
    render_overview()
with main_tabs[1]:
    render_performance_risks()
with main_tabs[2]:
    render_allocation_diversification()
with main_tabs[3]:
    render_strategy_dca()
with main_tabs[4]:
    render_monte_carlo()
with main_tabs[5]:
    render_ai_analysis()

# === FIN NOUVELLE STRUCTURE ===