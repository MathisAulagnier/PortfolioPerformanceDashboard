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
        # Récupération du taux des bons du Trésor 13 semaines (^IRX)
        treasury_data = yf.download("^IRX", period="5d", progress=False)
        if not treasury_data.empty and 'Close' in treasury_data.columns:
            # Le taux est donné en pourcentage, on le convertit en décimal
            latest_rate = treasury_data['Close'].iloc[-1] / 100.0
            return float(latest_rate)
        else:
            return 0.0
    except Exception:
        # En cas d'erreur, retourner 0
        return 0.0

def calculate_dca_portfolio(returns, weights, initial_capital, dca_enabled, dca_amount, dca_frequency):
    """
    Calcule l'évolution du portefeuille avec ou sans DCA (Dollar-Cost Averaging).
    
    Args:
        returns: DataFrame des rendements quotidiens des actifs
        weights: Liste des poids du portefeuille
        initial_capital: Capital initial
        dca_enabled: Boolean, True si DCA activé
        dca_amount: Montant à ajouter à chaque période DCA
        dca_frequency: 'Mensuelle' ou 'Annuelle'
    
    Returns:
        tuple: (portfolio_value_series, total_invested_series, portfolio_returns_series)
    """
    if returns.empty:
        return pd.Series(), pd.Series(), pd.Series()
    
    # Calcul des rendements du portefeuille pondéré
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # Initialisation des séries
    portfolio_value = pd.Series(index=returns.index, dtype=float)
    total_invested = pd.Series(index=returns.index, dtype=float)
    
    # Valeurs initiales
    current_value = initial_capital
    cumulative_invested = initial_capital
    
    portfolio_value.iloc[0] = current_value
    total_invested.iloc[0] = cumulative_invested
    
    # Configuration des dates DCA
    if dca_enabled:
        dca_dates = set()
        current_date = returns.index[0]
        end_date = returns.index[-1]
        
        while current_date <= end_date:
            # Trouver la prochaine date de marché disponible
            available_dates = returns.index[returns.index >= current_date]
            if len(available_dates) > 0:
                dca_dates.add(available_dates[0])
            
            # Incrémenter selon la fréquence
            if dca_frequency == 'Mensuelle':
                current_date += relativedelta(months=1)
            else:  # Annuelle
                current_date += relativedelta(years=1)
    
    # Calcul jour par jour
    for i in range(1, len(returns)):
        date = returns.index[i]
        daily_return = portfolio_returns.iloc[i]
        
        # Ajout DCA si c'est une date programmée
        if dca_enabled and date in dca_dates:
            current_value += dca_amount
            cumulative_invested += dca_amount
        
        # Application du rendement quotidien
        current_value *= (1 + daily_return)
        
        portfolio_value.iloc[i] = current_value
        total_invested.iloc[i] = cumulative_invested
    
    return portfolio_value, total_invested, portfolio_returns

def calculate_metrics_with_dca(portfolio_returns, portfolio_value, total_invested, risk_free_rate=0.0):
    """
    Calcule les métriques de performance en tenant compte des apports DCA.
    
    Args:
        portfolio_returns: Série des rendements quotidiens du portefeuille
        portfolio_value: Série de la valeur du portefeuille
        total_invested: Série du capital total investi
        risk_free_rate: Taux sans risque annuel (défaut: 0%)
    
    Returns:
        tuple: (rendement_total, rendement_annualisé, volatilité, ratio_sharpe, twr)
    """
    if portfolio_returns.empty or portfolio_returns.isnull().all():
        return 0, 0, 0, 0, 0
    
    # Rendement simple basé sur la valeur finale vs capital total investi
    simple_return = (portfolio_value.iloc[-1] / total_invested.iloc[-1]) - 1
    
    # Time-Weighted Return (TWR) - plus approprié pour les apports multiples
    twr = (1 + portfolio_returns).prod() - 1
    
    # Nombre de jours pour l'annualisation
    num_days = len(portfolio_returns)
    if num_days == 0:
        return 0, 0, 0, 0, 0
    
    # Rendement annualisé basé sur le TWR
    annualized_return = ((1 + twr) ** (252 / num_days)) - 1
    
    # Volatilité annualisée
    volatility = portfolio_returns.std() * np.sqrt(252)
    
    # Ratio de Sharpe (utilise le taux sans risque configuré)
    sharpe_ratio = (annualized_return - risk_free_rate) / (volatility + 1e-10)
    
    return simple_return, annualized_return, volatility, sharpe_ratio, twr

def calculate_advanced_metrics(portfolio_returns, benchmark_returns, risk_free_rate=0.0):
    """
    Calcule les métriques de performance avancées.
    
    Args:
        portfolio_returns: Série des rendements quotidiens du portefeuille
        benchmark_returns: Série des rendements quotidiens du benchmark
        risk_free_rate: Taux sans risque annuel (défaut: 0%)
    
    Returns:
        dict: Dictionnaire contenant sortino_ratio, alpha, beta
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        return {'sortino_ratio': 0, 'alpha': 0, 'beta': 0}
    
    # Alignement des séries temporelles
    aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner').dropna()
    if aligned_data.empty or len(aligned_data.columns) < 2:
        return {'sortino_ratio': 0, 'alpha': 0, 'beta': 0}
    
    portfolio_aligned = aligned_data.iloc[:, 0]
    benchmark_aligned = aligned_data.iloc[:, 1]
    
    # Taux sans risque quotidien
    daily_risk_free = risk_free_rate / 252
    
    # --- RATIO DE SORTINO ---
    # Calcul de la "downside deviation" (volatilité des rendements négatifs uniquement)
    negative_returns = portfolio_aligned[portfolio_aligned < daily_risk_free] - daily_risk_free
    if len(negative_returns) > 0:
        downside_deviation = negative_returns.std() * np.sqrt(252)
    else:
        downside_deviation = 1e-10  # Éviter division par zéro
    
    # Rendement annualisé du portefeuille
    portfolio_annual_return = ((1 + portfolio_aligned.mean()) ** 252) - 1
    sortino_ratio = (portfolio_annual_return - risk_free_rate) / downside_deviation
    
    # --- ALPHA ET BÊTA ---
    # Calcul du bêta par régression linéaire
    if benchmark_aligned.var() > 1e-10:  # Éviter division par zéro
        covariance = np.cov(portfolio_aligned, benchmark_aligned)[0, 1]
        benchmark_variance = benchmark_aligned.var()
        beta = covariance / benchmark_variance
    else:
        beta = 0
    
    # Calcul de l'alpha (excès de rendement non expliqué par le marché)
    benchmark_annual_return = ((1 + benchmark_aligned.mean()) ** 252) - 1
    alpha = portfolio_annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
    
    return {
        'sortino_ratio': sortino_ratio,
        'alpha': alpha,
        'beta': beta
    }

def calculate_drawdown_series(portfolio_value):
    """
    Calcule la série temporelle des drawdowns.
    
    Args:
        portfolio_value: Série de la valeur du portefeuille
    
    Returns:
        pd.Series: Série des drawdowns en pourcentage
    """
    if portfolio_value.empty or portfolio_value.isnull().all():
        return pd.Series()
    
    # Calcul du peak cumulé (plus haut historique)
    peak = portfolio_value.expanding(min_periods=1).max()
    
    # Calcul du drawdown à chaque instant
    drawdown = (portfolio_value / peak) - 1
    
    return drawdown

def calculate_max_drawdown(portfolio_value):
    """
    Calcule le Drawdown Maximal basé sur la valeur du portefeuille.
    """
    if portfolio_value.empty or portfolio_value.isnull().all():
        return 0
    
    # Utilise la fonction drawdown_series pour la cohérence
    drawdown_series = calculate_drawdown_series(portfolio_value)
    return drawdown_series.min() if not drawdown_series.empty else 0

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

st.title("Dashboard de Backtesting de Portefeuille")

if not st.session_state.tickers_list:
    st.info("Bienvenue ! Veuillez commencer par ajouter au moins une action dans la barre latérale.")
    st.stop()

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
    st.warning("Aucune donnée disponible pour les actions sélectionnées sur la période choisie. Essayez une période plus courte ou d'autres actions.")
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

# --- Graphique d'évolution du capital ---
st.subheader(f"Évolution du capital sur : **{selected_period}**")

# Ajout d'informations sur le DCA si activé
if dca_enabled:
    total_dca_added = total_invested.iloc[-1] - initial_capital
    st.info(f"**DCA activé** : {total_dca_added:,.0f}$ ajoutés sur la période (en plus du capital initial)")

fig = go.Figure()

# Ligne du portefeuille
fig.add_trace(go.Scatter(
    x=portfolio_value.index, 
    y=portfolio_value, 
    mode='lines', 
    name='Portefeuille', 
    line=dict(color='royalblue', width=2)
))

# Ligne du benchmark
fig.add_trace(go.Scatter(
    x=benchmark_value.index, 
    y=benchmark_value, 
    mode='lines', 
    name=f'Indice de référence ({benchmark})', 
    line=dict(color='grey', width=2, dash='dash')
))

# Si DCA activé, ajouter la ligne du capital investi
if dca_enabled:
    fig.add_trace(go.Scatter(
        x=total_invested.index, 
        y=total_invested, 
        mode='lines', 
        name='Capital Total Investi', 
        line=dict(color='orange', width=1, dash='dot')
    ))

fig.update_layout(
    title="Évolution comparative des investissements",
    xaxis_title="Date",
    yaxis_title="Valeur ($)",
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# --- NOUVEAU : Graphique du Drawdown ---
st.subheader("Évolution du Drawdown")

# Calcul des séries de drawdown
portfolio_drawdown = calculate_drawdown_series(portfolio_value)
benchmark_drawdown = calculate_drawdown_series(benchmark_value)

if not portfolio_drawdown.empty:
    fig_dd = go.Figure()
    
    # Drawdown du portefeuille
    fig_dd.add_trace(go.Scatter(
        x=portfolio_drawdown.index,
        y=portfolio_drawdown * 100,  # Conversion en pourcentage
        mode='lines',
        name='Drawdown Portefeuille',
        line=dict(color='red', width=2),
        fill='tonexty',  # Remplissage vers l'axe x
        fillcolor='rgba(255, 0, 0, 0.1)'
    ))
    
    # Drawdown du benchmark
    if not benchmark_drawdown.empty:
        fig_dd.add_trace(go.Scatter(
            x=benchmark_drawdown.index,
            y=benchmark_drawdown * 100,  # Conversion en pourcentage
            mode='lines',
            name=f'Drawdown {benchmark}',
            line=dict(color='grey', width=2, dash='dash'),
            fill='tonexty',  # Remplissage vers l'axe x
            fillcolor='rgba(128, 128, 128, 0.1)'
        ))
    
    # Ligne de référence à 0%
    fig_dd.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, opacity=0.5)
    
    # Marquage des drawdowns significatifs (> 10%)
    significant_dd = portfolio_drawdown[portfolio_drawdown < -0.1]
    if not significant_dd.empty:
        fig_dd.add_trace(go.Scatter(
            x=significant_dd.index,
            y=significant_dd * 100,
            mode='markers',
            name='Drawdowns > 10%',
            marker=dict(color='darkred', size=6, symbol='triangle-down'),
            hovertemplate='Date: %{x}<br>Drawdown: %{y:.1f}%<extra></extra>'
        ))
    
    fig_dd.update_layout(
        title="Évolution du Drawdown au fil du temps",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        yaxis=dict(
            tickformat='.1f',
            range=[min(portfolio_drawdown.min() * 100, benchmark_drawdown.min() * 100) * 1.1, 5]
        ),
        showlegend=True
    )
    
    # Informations contextuelles
    max_dd_date = portfolio_drawdown.idxmin()
    max_dd_value = portfolio_drawdown.min()
    
    # Calcul de la durée de récupération du drawdown maximal
    recovery_info = ""
    if max_dd_value < -0.01:  # Si drawdown > 1%
        # Trouver quand le portefeuille a retrouvé son niveau d'avant le drawdown max
        peak_before_dd = portfolio_value[:max_dd_date].max()
        recovery_dates = portfolio_value[max_dd_date:][portfolio_value[max_dd_date:] >= peak_before_dd]
        
        if not recovery_dates.empty:
            recovery_date = recovery_dates.index[0]
            recovery_days = (recovery_date - max_dd_date).days
            recovery_info = f" | Récupération: {recovery_days} jours"
    
    st.info(f"**Drawdown maximal**: {max_dd_value:.2%} le {max_dd_date.strftime('%d/%m/%Y')}{recovery_info}")
    
    st.plotly_chart(fig_dd, use_container_width=True)
    
    # Statistiques de drawdown
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_dd = portfolio_drawdown[portfolio_drawdown < 0].mean() if (portfolio_drawdown < 0).any() else 0
        st.metric("Drawdown Moyen", f"{avg_dd:.2%}", help="Drawdown moyen lors des périodes de baisse")
    
    with col2:
        dd_periods = (portfolio_drawdown < -0.05).sum()  # Nombre de jours avec DD > 5%
        st.metric("Jours DD > 5%", f"{dd_periods}", help="Nombre de jours avec drawdown supérieur à 5%")
    
    with col3:
        # Calcul du temps moyen de récupération
        dd_start_dates = portfolio_drawdown[(portfolio_drawdown.shift(1) >= 0) & (portfolio_drawdown < 0)].index
        recovery_times = []
        
        for start_date in dd_start_dates:
            peak_value = portfolio_value[start_date]
            future_values = portfolio_value[start_date:]
            recovery_idx = future_values[future_values >= peak_value]
            if not recovery_idx.empty:
                recovery_date = recovery_idx.index[0]
                recovery_times.append((recovery_date - start_date).days)
        
        avg_recovery = np.mean(recovery_times) if recovery_times else 0
        st.metric("Récupération Moy.", f"{avg_recovery:.0f}j", help="Temps moyen de récupération après un drawdown")
    
    with col4:
        # Pourcentage du temps en drawdown
        time_in_dd = (portfolio_drawdown < -0.01).sum() / len(portfolio_drawdown) * 100
        st.metric("Temps en DD", f"{time_in_dd:.1f}%", help="Pourcentage du temps passé en drawdown > 1%")

else:
    st.warning("Impossible de calculer le drawdown avec les données disponibles.")

# --- Affichage des métriques ---
st.subheader("Métriques de Performance")

# Information sur le taux sans risque utilisé
col_info1, col_info2 = st.columns([3, 1])
with col_info1:
    if use_auto_rate and auto_risk_free_rate > 0:
        st.info(f"**Taux sans risque utilisé**: {risk_free_rate:.2%} (Bons du Trésor US 13 semaines - automatique)")
    else:
        st.info(f"**Taux sans risque utilisé**: {risk_free_rate:.2%} (manuel)")
with col_info2:
    if st.button("🔄", help="Actualiser le taux sans risque automatique"):
        st.cache_data.clear()
        st.rerun()

# Calcul des métriques avec DCA
p_simple, p_annual, p_vol, p_sharpe, p_twr = calculate_metrics_with_dca(portfolio_returns, portfolio_value, total_invested, risk_free_rate)
p_drawdown = calculate_max_drawdown(portfolio_value)

# --- CORRECTION : Calcul des métriques benchmark avec DCA ---
if dca_enabled:
    # Utiliser la même logique DCA pour le benchmark
    benchmark_simple = (benchmark_value.iloc[-1] / benchmark_total_invested.iloc[-1]) - 1 if len(benchmark_value) > 0 else 0
    benchmark_twr = (1 + benchmark_returns).prod() - 1 if len(benchmark_returns) > 0 else 0
    num_days = len(benchmark_returns)
    b_annual = ((1 + benchmark_twr) ** (252 / num_days)) - 1 if num_days > 0 else 0
    b_total = benchmark_simple  # Pour l'affichage, on utilise le rendement vs capital investi
else:
    # Calcul standard sans DCA
    b_total = (benchmark_value.iloc[-1] / initial_capital) - 1 if len(benchmark_value) > 0 else 0
    num_days = len(benchmark_returns)
    b_annual = ((1 + b_total) ** (252 / num_days)) - 1 if num_days > 0 else 0

b_vol = benchmark_returns.std() * np.sqrt(252) if len(benchmark_returns) > 0 else 0
b_sharpe = (b_annual - risk_free_rate) / (b_vol + 1e-10)
b_drawdown = calculate_max_drawdown(benchmark_value)

# --- NOUVEAU : Calcul des métriques avancées ---
advanced_metrics = calculate_advanced_metrics(portfolio_returns, benchmark_returns, risk_free_rate)

# --- Métriques de base ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Portefeuille")
    
    if dca_enabled:
        st.metric("Rendement vs Capital Investi", f"{p_simple:.2%}", 
                 help="Rendement basé sur le capital total investi (DCA inclus)")
        st.metric("Time-Weighted Return", f"{p_twr:.2%}", 
                 help="Rendement pondéré dans le temps (indépendant des apports)")
    else:
        st.metric("Rendement Total", f"{p_simple:.2%}")
    
    st.metric("Rendement Annualisé", f"{p_annual:.2%}")
    st.metric("Volatilité", f"{p_vol:.2%}")
    st.metric("Ratio de Sharpe", f"{p_sharpe:.2f}", help=f"Ratio de Sharpe calculé avec un taux sans risque de {risk_free_rate:.2%}")
    st.metric("Drawdown Maximal", f"{p_drawdown:.2%}", delta_color="inverse")

with col2:
    st.markdown("#### Indice de référence")
    
    if dca_enabled:
        st.metric("Rendement vs Capital Investi", f"{b_total:.2%}", 
                 help="Rendement de l'indice de référence avec les mêmes apports DCA")
    else:
        st.metric("Rendement Total", f"{b_total:.2%}")
    
    st.metric("Rendement Annualisé", f"{b_annual:.2%}")
    st.metric("Volatilité", f"{b_vol:.2%}")
    st.metric("Ratio de Sharpe", f"{b_sharpe:.2f}", help=f"Ratio de Sharpe calculé avec un taux sans risque de {risk_free_rate:.2%}")
    st.metric("Drawdown Maximal", f"{b_drawdown:.2%}", delta_color="inverse")

# --- NOUVEAU : Section Métriques Avancées ---
st.markdown("---")
st.subheader("Métriques de Performance Avancées")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Ratio de Sortino")
    st.metric(
        "Sortino", 
        f"{advanced_metrics['sortino_ratio']:.2f}",
        help=f"Ratio de Sharpe modifié qui ne pénalise que la volatilité négative. Calculé avec un taux sans risque de {risk_free_rate:.2%}. Plus élevé = meilleur."
    )
    
    # Interprétation du Sortino
    sortino_val = advanced_metrics['sortino_ratio']
    if sortino_val > 2:
        st.success("Excellent (> 2.0)")
    elif sortino_val > 1:
        st.info("Bon (1.0 - 2.0)")
    elif sortino_val > 0:
        st.warning("Correct (0 - 1.0)")
    else:
        st.error("Faible (< 0)")

with col2:
    st.markdown("#### Bêta vs Indice de référence")
    st.metric(
        "Bêta (β)", 
        f"{advanced_metrics['beta']:.2f}",
        help="Sensibilité aux mouvements du marché. β=1: suit le marché, β>1: amplifie, β<1: atténue."
    )
    
    # Interprétation du Bêta
    beta_val = advanced_metrics['beta']
    if beta_val > 1.2:
        st.info("Très volatil (β > 1.2)")
    elif beta_val > 1:
        st.info("Plus volatil que le marché")
    elif beta_val > 0.8:
        st.success("Proche du marché")
    elif beta_val > 0:
        st.success("Moins volatil que le marché")
    else:
        st.warning("Corrélation négative")

with col3:
    st.markdown("#### Alpha (Surperformance)")
    st.metric(
        "Alpha (α)", 
        f"{advanced_metrics['alpha']:.2%}",
        help=f"Rendement excédentaire non expliqué par le marché. Calculé avec un taux sans risque de {risk_free_rate:.2%}. Positif = surperformance."
    )
    
    # Interprétation de l'Alpha
    alpha_val = advanced_metrics['alpha']
    if alpha_val > 0.05:
        st.success("Forte surperformance (> 5%)")
    elif alpha_val > 0.02:
        st.success("Surperformance (2-5%)")
    elif alpha_val > -0.02:
        st.info("Performance neutre (±2%)")
    else:
        st.error("Sous-performance (< -2%)")

# Explication des métriques avancées
with st.expander("Comprendre les métriques avancées"):
    st.markdown(f"""
    **Ratio de Sortino** :
    - Amélioration du ratio de Sharpe qui ne pénalise que la "mauvaise" volatilité
    - Ne considère que les écarts négatifs par rapport au rendement attendu
    - Plus approprié pour les investisseurs averses aux pertes
    - **Taux sans risque utilisé**: {risk_free_rate:.2%}
    
    **Bêta (β)** :
    - Mesure la sensibilité de votre portefeuille aux mouvements de l'indice de référence
    - β = 1 : Le portefeuille suit exactement le marché
    - β > 1 : Le portefeuille amplifie les mouvements (plus risqué)
    - β < 1 : Le portefeuille atténue les mouvements (plus défensif)
    
    **Alpha (α)** :
    - Mesure la valeur ajoutée de votre stratégie d'investissement
    - Rendement excédentaire après ajustement pour le risque systématique
    - α > 0 : Votre stratégie bat le marché ajusté du risque
    - α < 0 : Votre stratégie sous-performe le marché ajusté du risque
    - **Taux sans risque utilisé**: {risk_free_rate:.2%}
    
    **Note sur le taux sans risque** :
    - Le taux sans risque est automatiquement récupéré des bons du Trésor américain (13 semaines)
    - Vous pouvez choisir d'utiliser un taux manuel dans la barre latérale
    - Ce taux impacte directement les calculs de Sharpe, Sortino et Alpha
    """)

# --- Graphiques des métriques avancées ---
st.markdown("---")
st.subheader("Analyse de Régression Portfolio vs Indice de référence")

# Graphique de corrélation
if not portfolio_returns.empty and not benchmark_returns.empty:
    # Alignement des données pour le graphique
    aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner').dropna()
    if not aligned_data.empty and len(aligned_data.columns) >= 2:
        portfolio_aligned = aligned_data.iloc[:, 0] * 100  # Conversion en %
        benchmark_aligned = aligned_data.iloc[:, 1] * 100   # Conversion en %
        
        # Création du graphique de régression
        fig_regression = go.Figure()
        
        # Scatter plot des rendements
        fig_regression.add_trace(go.Scatter(
            x=benchmark_aligned,
            y=portfolio_aligned,
            mode='markers',
            name='Rendements quotidiens',
            marker=dict(
                color='royalblue',
                size=4,
                opacity=0.6
            ),
            hovertemplate='Benchmark: %{x:.2f}%<br>Portefeuille: %{y:.2f}%<extra></extra>'
        ))
        
        # Ligne de régression
        if len(benchmark_aligned) > 1:
            slope, intercept = np.polyfit(benchmark_aligned, portfolio_aligned, 1)
            line_x = np.array([benchmark_aligned.min(), benchmark_aligned.max()])
            line_y = slope * line_x + intercept
            
            fig_regression.add_trace(go.Scatter(
                x=line_x,
                y=line_y,
                mode='lines',
                name=f'Régression (β={slope:.2f})',
                line=dict(color='red', width=2)
            ))
        
        fig_regression.update_layout(
            title=f"Corrélation Portfolio vs {benchmark} (β = {advanced_metrics['beta']:.2f})",
            xaxis_title=f"Rendements {benchmark} (%)",
            yaxis_title="Rendements Portfolio (%)",
            hovermode='closest'
        )
        
        st.plotly_chart(fig_regression, use_container_width=True)
    else:
        st.info("Données insuffisantes pour créer le graphique de régression.")
else:
    st.info("Données insuffisantes pour l'analyse de régression.")

# --- Affichage des statistiques DCA ---
if dca_enabled:
    st.markdown("---")
    st.subheader("Statistiques DCA")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Capital Initial", f"{initial_capital:,.0f}$")
    
    with col2:
        total_dca = total_invested.iloc[-1] - initial_capital
        st.metric("Total Ajouté (DCA)", f"{total_dca:,.0f}$")
    
    with col3:
        st.metric("Capital Total Investi", f"{total_invested.iloc[-1]:,.0f}$")

# --- Répartition et Exposition Géographique ---
st.subheader("Répartition et Exposition Géographique")

# Récupération des informations géographiques
geo_data = []
for ticker in valid_tickers:
    try:
        ticker_info = yf.Ticker(ticker).info
        country = ticker_info.get('country', 'Inconnu')
        geo_data.append({'Ticker': ticker, 'Country': country, 'Weight': st.session_state.weights.get(ticker, 0)})
    except Exception:
        pass

# Regroupement par pays
if geo_data:
    geo_df = pd.DataFrame(geo_data)
    geo_df = geo_df.groupby('Country').agg({'Weight': 'sum'}).reset_index()
    geo_df = geo_df[geo_df['Weight'] > 0]
else:
    geo_df = pd.DataFrame()

# Affichage des graphiques géographiques
if not geo_df.empty:
    col1, col2 = st.columns(2)

    with col1:
        fig_geo = px.pie(
            geo_df,
            names='Country',
            values='Weight',
            title='Répartition par Pays'
        )
        fig_geo.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_geo, use_container_width=True)

    with col2:
        # Liste des pays en situation de conflit/guerre (à adapter selon l'actualité)
        conflict_countries = ['Ukraine', 'Russia', 'Afghanistan', 'Syria', 'Yemen', 'Myanmar', 'Ethiopia', 'Sudan', 'Gaza', 'Israel', 'Palestine']
        
        fig_map = px.choropleth(
            geo_df,
            locations='Country',
            locationmode='country names',
            hover_name='Country',
            title='Exposition sur la Carte',
            color='Weight',
            color_continuous_scale=px.colors.sequential.Plasma,
            labels={'Weight': 'Poids (%)'}
        )
        
        # Ajouter les contours noirs pour les pays en conflit
        fig_map.add_trace(go.Choropleth(
            locations=conflict_countries,
            locationmode='country names',
            z=[1] * len(conflict_countries),  # Valeur arbitraire
            colorscale=[[0, 'black'], [1, 'black']],  # Couleur noire
            showscale=False,  # Pas d'échelle de couleur
            hovertemplate='<b>%{location}</b><br>Zone de conflit<extra></extra>',
            name='Zones de conflit',
            marker_line_color='red',  # Contour rouge pour plus de visibilité
            marker_line_width=2
        ))
        
        fig_map.update_geos(fitbounds="locations", visible=True)
        st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("Aucune donnée géographique disponible pour créer les graphiques.")

# --- Répartition Sectorielle ---
st.subheader("Répartition par Secteurs")

# Récupération des informations sectorielles
sector_data = []
for ticker in valid_tickers:
    try:
        ticker_info = yf.Ticker(ticker).info
        sector = ticker_info.get('sectorKey', 'Inconnu')
        weight = st.session_state.weights.get(ticker, 0)
        sector_data.append({
            'Ticker': ticker, 
            'Sector': sector, 
            'Weight': weight
        })
    except Exception as e:
        # En cas d'erreur, ajouter avec secteur inconnu
        sector_data.append({
            'Ticker': ticker, 
            'Sector': 'Inconnu', 
            'Weight': st.session_state.weights.get(ticker, 0)
        })

# Regroupement par secteur
if sector_data:
    sector_df = pd.DataFrame(sector_data)
    sector_summary = sector_df.groupby('Sector').agg({
        'Weight': 'sum',
        'Ticker': lambda x: ', '.join(x)
    }).reset_index()
    sector_summary = sector_summary[sector_summary['Weight'] > 0]
    sector_summary.columns = ['Secteur', 'Poids (%)', 'Tickers']
else:
    sector_summary = pd.DataFrame()

# Récupération des informations industrielles
industry_data = []
for ticker in valid_tickers:
    try:
        ticker_info = yf.Ticker(ticker).info
        industry = ticker_info.get('industryKey', 'Inconnu')
        weight = st.session_state.weights.get(ticker, 0)
        industry_data.append({
            'Ticker': ticker, 
            'Industry': industry, 
            'Weight': weight
        })
    except Exception as e:
        # En cas d'erreur, ajouter avec industrie inconnue
        industry_data.append({
            'Ticker': ticker, 
            'Industry': 'Inconnu', 
            'Weight': st.session_state.weights.get(ticker, 0)
        })
# Regroupement par industrie
if industry_data:
    industry_df = pd.DataFrame(industry_data)
    industry_summary = industry_df.groupby('Industry').agg({
        'Weight': 'sum',
        'Ticker': lambda x: ', '.join(x)
    }).reset_index()
    industry_summary = industry_summary[industry_summary['Weight'] > 0]
    industry_summary.columns = ['Industrie', 'Poids (%)', 'Tickers']

# Affichage des graphiques sectoriels et industriels
if not sector_summary.empty or not industry_summary.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique en secteurs (pie chart)
        fig_sector = px.pie(
            sector_summary,
            names='Secteur',
            values='Poids (%)',
            title='Répartition par Secteurs',
            hover_data=['Tickers'],
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_sector.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>' +
                         'Poids: %{value}%<br>' +
                         'Tickers: %{customdata[0]}<br>' +
                         '<extra></extra>'
        )
        st.plotly_chart(fig_sector, use_container_width=True)
    
    with col2:
        # Graphique en industries (pie chart)
        fig_industry = px.pie(
            industry_summary,
            names='Industrie',
            values='Poids (%)',
            title='Répartition par Industries',
            hover_data=['Tickers'],
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_industry.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>' +
                            'Poids: %{value}%<br>' +
                            'Tickers: %{customdata[0]}<br>' +
                            '<extra></extra>'
        )
        st.plotly_chart(fig_industry, use_container_width=True)
else:
    st.info("Aucune donnée sectorielle disponible pour créer les graphiques.")
    
    # Tableau détaillé des secteurs
    st.markdown("#### Détail par Secteur")
    
    # Formatage du tableau pour l'affichage
    display_df = sector_summary.copy()
    display_df['Poids (%)'] = display_df['Poids (%)'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Secteur": st.column_config.TextColumn("Secteur", width="medium"),
            "Poids (%)": st.column_config.TextColumn("Poids", width="small"),
            "Tickers": st.column_config.TextColumn("Actions", width="large")
        }
    )
    
    # Analyse de diversification sectorielle
    st.markdown("#### Analyse de la Diversification Sectorielle")
    
    num_sectors = len(sector_summary)
    max_sector_weight = sector_summary['Poids (%)'].max()
    dominant_sector = sector_summary.loc[sector_summary['Poids (%)'].idxmax(), 'Secteur']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nombre de Secteurs", num_sectors)
    
    with col2:
        st.metric("Secteur Dominant", dominant_sector)
    
    with col3:
        st.metric("Poids Max", f"{max_sector_weight:.1f}%")
    
    with col4:
        # Calcul de l'indice de concentration (Herfindahl)
        hhi = sum((weight/100)**2 for weight in sector_summary['Poids (%)'])
        st.metric("Indice HHI", f"{hhi:.3f}", help="Indice de Herfindahl-Hirschman. Plus proche de 0 = plus diversifié")
    
    # Conseils de diversification
    if num_sectors < 3:
        st.warning("⚠️ **Diversification faible** : Votre portefeuille est concentré sur moins de 3 secteurs. Considérez diversifier davantage.")
    elif max_sector_weight > 50:
        st.warning(f"⚠️ **Concentration élevée** : Le secteur '{dominant_sector}' représente {max_sector_weight:.1f}% du portefeuille.")
    elif num_sectors >= 5 and max_sector_weight < 40:
        st.success("✅ **Bonne diversification** : Votre portefeuille est bien réparti entre plusieurs secteurs.")
    else:
        st.info("ℹ️ **Diversification correcte** : Répartition sectorielle équilibrée.")
    
    
    # Classification des secteurs par type
    defensive_sectors = ['Utilities', 'Consumer Staples', 'Healthcare', 'Real Estate']
    cyclical_sectors = ['Consumer Discretionary', 'Industrials', 'Materials', 'Energy']
    growth_sectors = ['Technology', 'Communication Services']
    financial_sectors = ['Financial Services', 'Financials']
    
    sector_types = {'Défensifs': 0, 'Cycliques': 0, 'Croissance': 0, 'Financiers': 0, 'Autres': 0}
    
    for _, row in sector_summary.iterrows():
        sector = row['Secteur']
        weight = row['Poids (%)']
        
        if sector in defensive_sectors:
            sector_types['Défensifs'] += weight
        elif sector in cyclical_sectors:
            sector_types['Cycliques'] += weight
        elif sector in growth_sectors:
            sector_types['Croissance'] += weight
        elif sector in financial_sectors:
            sector_types['Financiers'] += weight
        else:
            sector_types['Autres'] += weight
    

#####

# --- Fonction d'analyse par IA ---
def generate_portfolio_analysis(portfolio_data, benchmark_data, metrics, advanced_metrics, period, tickers_list, weights, 
                               additional_context=None):
    """
    Génère une analyse détaillée du portefeuille via OpenAI GPT.
    
    Args:
        portfolio_data: Données de performance du portefeuille
        benchmark_data: Données de performance du benchmark
        metrics: Métriques de base (rendement, volatilité, etc.)
        advanced_metrics: Métriques avancées (alpha, beta, sortino)
        period: Période d'analyse sélectionnée
        tickers_list: Liste des tickers du portefeuille
        weights: Dictionnaire des poids du portefeuille
        additional_context: Contexte supplémentaire (DCA, secteurs, géographie, etc.)
    
    Returns:
        str: Analyse textuelle générée par l'IA
    """
    try:
        # Configuration du client OpenAI (nouvelle API)
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        # Récupération d'informations détaillées sur les actifs
        assets_details = {}
        total_market_cap = 0
        for ticker in tickers_list:
            try:
                ticker_info = yf.Ticker(ticker).info
                market_cap = ticker_info.get('marketCap', 0)
                total_market_cap += market_cap
                
                assets_details[ticker] = {
                    "poids": weights.get(ticker, 0),
                    "secteur": ticker_info.get('sectorKey', 'Inconnu'),
                    "industrie": ticker_info.get('industryKey', 'Inconnu'),
                    "pays": ticker_info.get('country', 'Inconnu'),
                    "capitalisation": market_cap,
                    "pe_ratio": ticker_info.get('trailingPE', 'N/A'),
                    "dividend_yield": ticker_info.get('dividendYield', 0),
                    "beta_individuel": ticker_info.get('beta', 'N/A'),
                    "nom_complet": ticker_info.get('longName', ticker)
                }
            except:
                assets_details[ticker] = {
                    "poids": weights.get(ticker, 0),
                    "secteur": "Inconnu",
                    "industrie": "Inconnu", 
                    "pays": "Inconnu",
                    "capitalisation": 0,
                    "pe_ratio": "N/A",
                    "dividend_yield": 0,
                    "beta_individuel": "N/A",
                    "nom_complet": ticker
                }
        
        # Analyse sectorielle et géographique
        secteurs = {}
        pays = {}
        for ticker, details in assets_details.items():
            secteur = details["secteur"]
            pays_actif = details["pays"]
            poids = details["poids"]
            
            secteurs[secteur] = secteurs.get(secteur, 0) + poids
            pays[pays_actif] = pays.get(pays_actif, 0) + poids
        
        # Statistiques de drawdown détaillées
        portfolio_drawdown = calculate_drawdown_series(portfolio_data)
        drawdown_stats = {}
        if not portfolio_drawdown.empty:
            drawdown_stats = {
                "drawdown_moyen": f"{portfolio_drawdown[portfolio_drawdown < 0].mean():.2%}" if (portfolio_drawdown < 0).any() else "0.00%",
                "jours_drawdown_5pc": (portfolio_drawdown < -0.05).sum(),
                "temps_en_drawdown": f"{(portfolio_drawdown < -0.01).sum() / len(portfolio_drawdown) * 100:.1f}%",
                "nombre_drawdowns_10pc": (portfolio_drawdown < -0.1).sum()
            }
        
        # Préparation des données enrichies pour l'IA
        analysis_data = {
            "contexte_general": {
                "periode_analysee": period,
                "nombre_actifs": len(tickers_list),
                "date_analyse": datetime.now().strftime("%Y-%m-%d"),
                "capitalisation_totale_approximative": f"{total_market_cap:,.0f}$" if total_market_cap > 0 else "Non disponible"
            },
            "portefeuille": {
                "composition": assets_details,
                "metriques_performance": {
                    "rendement_total": f"{metrics['portfolio_simple']:.2%}",
                    "rendement_annualise": f"{metrics['portfolio_annual']:.2%}",
                    "volatilite": f"{metrics['portfolio_vol']:.2%}",
                    "sharpe": f"{metrics['portfolio_sharpe']:.2f}",
                    "sortino": f"{advanced_metrics['sortino_ratio']:.2f}",
                    "alpha": f"{advanced_metrics['alpha']:.2%}",
                    "beta": f"{advanced_metrics['beta']:.2f}",
                    "drawdown_max": f"{metrics['portfolio_drawdown']:.2%}"
                },
                "diversification": {
                    "repartition_sectorielle": secteurs,
                    "repartition_geographique": pays,
                    "concentration_max": f"{max(weights.values()):.1f}%" if weights else "0%",
                    "nombre_secteurs": len([s for s in secteurs.keys() if s != "Inconnu"]),
                    "nombre_pays": len([p for p in pays.keys() if p != "Inconnu"])
                },
                "statistiques_drawdown": drawdown_stats
            },
            "benchmark": {
                "type": additional_context.get("benchmark_name", "Indice de référence") if additional_context else "Indice de référence",
                "metriques": {
                    "rendement_total": f"{metrics['benchmark_total']:.2%}",
                    "rendement_annualise": f"{metrics['benchmark_annual']:.2%}",
                    "volatilite": f"{metrics['benchmark_vol']:.2%}",
                    "sharpe": f"{metrics['benchmark_sharpe']:.2f}",
                    "drawdown_max": f"{metrics['benchmark_drawdown']:.2%}"
                }
            }
        }
        
        # Ajout du contexte DCA si disponible
        if additional_context and additional_context.get("dca_enabled"):
            analysis_data["strategie_investissement"] = {
                "dca_active": True,
                "frequence": additional_context.get("dca_frequency", "Mensuelle"),
                "montant_periodique": f"{additional_context.get('dca_amount', 0):,.0f}$",
                "capital_initial": f"{additional_context.get('initial_capital', 0):,.0f}$",
                "total_investi": f"{additional_context.get('total_invested', 0):,.0f}$",
                "apports_supplementaires": f"{additional_context.get('total_dca_added', 0):,.0f}$"
            }
        else:
            analysis_data["strategie_investissement"] = {
                "dca_active": False,
                "capital_initial": f"{additional_context.get('initial_capital', 0):,.0f}$" if additional_context else "Non spécifié"
            }
        
        # Ajout du taux sans risque utilisé
        if additional_context and 'risk_free_rate' in additional_context:
            analysis_data["parametres_calcul"] = {
                "taux_sans_risque": f"{additional_context['risk_free_rate']:.2%}",
                "source_taux": additional_context.get("risk_free_source", "Manuel")
            }
        
        # Prompt détaillé pour l'analyse
        prompt = f"""
En tant qu'analyste financier expert, analysez ce portefeuille d'investissement en utilisant toutes les données détaillées fournies.

DONNÉES COMPLÈTES DU PORTEFEUILLE:
{json.dumps(analysis_data, indent=2, ensure_ascii=False)}

Utilisez ces données enrichies pour fournir une analyse approfondie et personnalisée couvrant:

1. **PERFORMANCE GLOBALE** (2-3 paragraphes)
   - Évaluation détaillée de la performance vs benchmark sur {period}
   - Analyse de l'efficacité de la stratégie (DCA, allocation, timing)
   - Points forts et faiblesses par rapport aux objectifs d'investissement

2. **ANALYSE DES RISQUES ET DIVERSIFICATION** (3-4 paragraphes)
   - Évaluation du couple risque/rendement avec métriques Sharpe et Sortino
   - Analyse approfondie des drawdowns (fréquence, durée, récupération)
   - Qualité de la diversification sectorielle et géographique
   - Concentration des positions et risques associés
   - Analyse du Bêta et de l'Alpha par rapport au benchmark

3. **COMPOSITION ET ALLOCATION** (2-3 paragraphes)
   - Analyse détaillée de chaque actif (secteur, pays, capitalisation, ratios)
   - Cohérence de l'allocation sectorielle et géographique
   - Équilibre entre croissance/défensif, large cap/small cap
   - Commentaires sur les ratios PE et rendements de dividendes

4. **STRATÉGIE D'INVESTISSEMENT** (2 paragraphes)
   - Évaluation de la stratégie DCA si applicable (efficacité, timing)
   - Analyse de l'utilisation du capital et des paramètres de risque
   - Adéquation avec le profil d'investisseur (horizon, tolérance au risque)

5. **RECOMMANDATIONS STRATÉGIQUES** (4-5 points concrets)
   - Suggestions d'amélioration de l'allocation sectorielle/géographique
   - Recommandations sur la gestion des risques et du drawdown
   - Optimisation de la stratégie DCA si applicable
   - Conseils pour améliorer le couple risque/rendement
   - Suggestions de rééquilibrage ou d'ajustements

6. **ALERTES ET SURVEILLANCE** (1-2 paragraphes)
   - Points de vigilance spécifiques aux secteurs/pays exposés
   - Métriques clés à surveiller régulièrement
   - Conditions de marché qui pourraient affecter le portefeuille

Utilisez les données détaillées sur chaque actif, la diversification, et les métriques de risque pour fournir des conseils précis et actionnables. Mentionnez spécifiquement les entreprises, secteurs et pays quand c'est pertinent.
"""

        # Appel à l'API OpenAI (nouvelle syntaxe)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Vous êtes un analyste financier expert spécialisé dans l'analyse de portefeuilles d'investissement. Vous fournissez des analyses détaillées, objectives et des conseils pratiques basés sur les données quantitatives."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Erreur lors de la génération de l'analyse IA : {str(e)}\n\nVérifiez que votre clé API OpenAI est correctement configurée dans .streamlit/secrets.toml"

# --- Section à ajouter après les graphiques géographiques, juste avant la fin ---

# Analyse par Intelligence Artificielle
st.markdown("---")
st.header("Analyse Détaillée par Intelligence Artificielle")

# Bouton pour déclencher l'analyse
if st.button("Générer l'Analyse IA", type="primary", help="Génère une analyse détaillée de votre portefeuille via GPT-4"):
    
    with st.spinner("L'IA analyse votre portefeuille... (cela peut prendre 10-30 secondes)"):
        # Préparation des métriques pour l'IA
        metrics_for_ai = {
            'portfolio_simple': p_simple,
            'portfolio_annual': p_annual,
            'portfolio_vol': p_vol,
            'portfolio_sharpe': p_sharpe,
            'portfolio_drawdown': p_drawdown,
            'benchmark_total': b_total,
            'benchmark_annual': b_annual,
            'benchmark_vol': b_vol,
            'benchmark_sharpe': b_sharpe,
            'benchmark_drawdown': b_drawdown
        }
        
        # Préparation du contexte enrichi pour l'IA
        benchmark_names = {
            "^GSPC": "S&P 500", 
            "^IXIC": "NASDAQ", 
            "GC=F": "Or", 
            "DX-Y.NYB": "Dollar Index"
        }
        
        additional_context = {
            "benchmark_name": benchmark_names.get(benchmark, benchmark),
            "dca_enabled": dca_enabled,
            "dca_frequency": dca_frequency if dca_enabled else None,
            "dca_amount": dca_amount if dca_enabled else 0,
            "initial_capital": initial_capital,
            "total_invested": total_invested.iloc[-1] if dca_enabled and not total_invested.empty else initial_capital,
            "total_dca_added": (total_invested.iloc[-1] - initial_capital) if dca_enabled and not total_invested.empty else 0,
            "risk_free_rate": risk_free_rate,
            "risk_free_source": "Automatique (Bons du Trésor US)" if use_auto_rate and auto_risk_free_rate > 0 else "Manuel",
            "period_start": start_date.strftime("%Y-%m-%d"),
            "period_end": end_date.strftime("%Y-%m-%d"),
            "valid_tickers": valid_tickers,
            "drawdown_series": portfolio_drawdown if not portfolio_drawdown.empty else None
        }
        
        # Génération de l'analyse
        ai_analysis = generate_portfolio_analysis(
            portfolio_data=portfolio_value,
            benchmark_data=benchmark_value,
            metrics=metrics_for_ai,
            advanced_metrics=advanced_metrics,
            period=selected_period,
            tickers_list=st.session_state.tickers_list,
            weights=st.session_state.weights,
            additional_context=additional_context
        )
        
        # Affichage de l'analyse
        st.markdown("### Rapport d'Analyse Personnalisé")
        
        # Affichage direct du Markdown pour préserver le formatage
        with st.container():            
            # Affichage du contenu Markdown
            st.markdown(ai_analysis)
        
        # Options d'export
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            # Bouton de téléchargement du rapport
            st.download_button(
                label="Télécharger le Rapport",
                data=ai_analysis,
                file_name=f"analyse_portefeuille_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        with col2:
            # Option pour régénérer l'analyse
            if st.button("Nouvelle Analyse"):
                st.rerun()
        
        with col3:
            st.info("**Astuce**: Modifiez votre portefeuille ou la période pour obtenir de nouvelles perspectives !")


st.markdown("---")
# Section d'information sur l'IA
with st.expander("À propos de l'analyse IA"):
    st.markdown("""
    **Comment fonctionne l'analyse IA ?**
    
    L'analyse est générée par **GPT-4**, le modèle de langage avancé d'OpenAI, qui :
    - Analyse toutes vos métriques de performance
    - Compare votre portefeuille à l'indice de référence sélectionné
    - Identifie les forces et faiblesses de votre stratégie
    - Propose des recommandations personnalisées
                
    **Coût :**
    - Chaque analyse coûte environ 0,02-0,05$ en crédits OpenAI
    - Le coût dépend de la complexité de votre portefeuille
    """)

# Section d'aide pour la sauvegarde/chargement
with st.expander("Guide : Sauvegarde et Chargement de Portefeuilles"):
    st.markdown("""
    **Sauvegarde de votre portefeuille :**
    1. Configurez votre portefeuille (tickers et répartitions)
    2. Cliquez sur "Sauvegarder" dans la barre latérale
    3. Un fichier JSON sera téléchargé avec la configuration actuelle
    
    **Chargement d'un portefeuille :**
    1. Utilisez "Charger un Portefeuille" dans la barre latérale
    2. Sélectionnez un fichier JSON précédemment sauvegardé
    3. Le portefeuille sera automatiquement restauré
    
    **Avantages :**
    - **Comparaison de stratégies** : Sauvegardez différentes allocations pour les comparer
    - **Sauvegarde de sessions** : Reprenez votre travail là où vous l'aviez laissé
    - **Partage** : Partagez vos configurations avec d'autres utilisateurs
    - **Historique** : Gardez une trace de l'évolution de vos stratégies
    
    **Note :** Les fichiers de portefeuille contiennent uniquement les tickers et leurs poids, 
    pas les données de marché qui sont toujours récupérées en temps réel.
    """)
