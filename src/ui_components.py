# ui_components.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from dateutil.relativedelta import relativedelta
import json
import yfinance as yf

# Import des modules de logique métier
import data_manager
import calculations
import ai_analyzer

def render_sidebar():
    """Gère l'affichage et la logique de la barre latérale."""
    st.sidebar.title("Paramètres du Backtest")
    st.sidebar.markdown("Commencez par ajouter des actions à votre portefeuille.")

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
    
    # Section de gestion du portefeuille
    if st.session_state.tickers_list:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Portefeuille Actuel")
        st.session_state.weights = {ticker: st.session_state.weights.get(ticker, 0) for ticker in st.session_state.tickers_list}
        for ticker in st.session_state.tickers_list:
            st.sidebar.markdown(f"**{ticker}**")
        
        portfolio_data = {
            "tickers": st.session_state.tickers_list, "weights": st.session_state.weights,
            "saved_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "version": "1.0"
        }
        portfolio_json = json.dumps(portfolio_data, indent=2, ensure_ascii=False)
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.download_button(label="Sauvegarder", data=portfolio_json, file_name=f"portefeuille_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")
        with col2:
            if st.button("Vider", type="secondary"):
                st.session_state.tickers_list = []
                st.session_state.weights = {}
                st.rerun()

    # Section de chargement
    st.sidebar.markdown("---")
    st.sidebar.subheader("Charger un Portefeuille")
    uploaded_file = st.sidebar.file_uploader("Choisissez un fichier JSON", type=['json'], key="portfolio_uploader")
    
    if uploaded_file is not None and not st.session_state.portfolio_loaded:
        try:
            portfolio_data = json.load(uploaded_file)
            if "tickers" in portfolio_data and "weights" in portfolio_data:
                valid_tickers, valid_weights = [], {}
                for ticker in portfolio_data["tickers"]:
                    if not yf.Ticker(ticker).history(period="5d").empty:
                        valid_tickers.append(ticker)
                        valid_weights[ticker] = portfolio_data["weights"].get(ticker, 0)
                    else:
                        st.sidebar.warning(f"Ticker '{ticker}' ignoré (invalide).")
                
                if valid_tickers:
                    st.session_state.tickers_list, st.session_state.weights = valid_tickers, valid_weights
                    st.session_state.portfolio_loaded = True
                    st.sidebar.success(f"Portefeuille chargé !")
                    st.rerun()
                else:
                    st.sidebar.error("Aucun ticker valide dans le fichier.")
            else:
                st.sidebar.error("Format de fichier invalide.")
        except Exception as e:
            st.sidebar.error(f"Erreur de chargement: {e}")

    if uploaded_file is None and st.session_state.portfolio_loaded:
        st.session_state.portfolio_loaded = False

    # Section répartition
    if st.session_state.tickers_list:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Répartition du Portefeuille (%)")
        total_weight = 0
        for ticker in st.session_state.tickers_list:
            weight = st.sidebar.number_input(f"Poids pour {ticker}", min_value=0, max_value=100, value=st.session_state.weights.get(ticker, 0), step=5, key=f"weight_{ticker}")
            st.session_state.weights[ticker] = weight
            total_weight += weight
        st.sidebar.markdown(f"**Total alloué : {total_weight}%**")

    # --- Configuration du backtest ---
    config = {}
    st.sidebar.markdown("---")
    st.sidebar.subheader("Configuration du Backtest")
    
    # Période
    periods = {"1 an": 12, "3 ans": 36, "5 ans": 60, "10 ans": 120, "20 ans": 240, "Max": None}
    config['selected_period'] = st.sidebar.radio("Période", options=list(periods.keys()))
    config['end_date'] = datetime.now()
    config['start_date'] = config['end_date'] - relativedelta(months=periods[config['selected_period']]) if periods[config['selected_period']] is not None else datetime(1995, 1, 1)

    # Capital et Benchmark
    config['initial_capital'] = st.sidebar.number_input("Capital Initial ($)", min_value=1000, value=10000, step=1000)
    config['benchmark'] = st.sidebar.selectbox("Indice de référence", ["^GSPC", "^IXIC", "GC=F", "DX-Y.NYB"], format_func=lambda x: {"^GSPC": "S&P 500", "^IXIC": "NASDAQ", "GC=F": "Or", "DX-Y.NYB": "Dollar Index"}[x])

    # DCA
    config['dca_enabled'] = st.sidebar.checkbox("Activer l'investissement programmé (DCA)", value=False)
    if config['dca_enabled']:
        config['dca_frequency'] = st.sidebar.selectbox("Fréquence d'investissement", ["Mensuelle", "Annuelle"])
        config['dca_amount'] = st.sidebar.number_input("Montant à ajouter ($)", min_value=100, max_value=10000, value=1000, step=100)
    else:
        config['dca_frequency'], config['dca_amount'] = "Mensuelle", 0

    # Taux sans risque
    st.sidebar.markdown("---")
    st.sidebar.subheader("Taux Sans Risque")
    auto_risk_free_rate = data_manager.get_risk_free_rate()
    use_auto_rate = st.sidebar.checkbox("Utiliser le taux automatique", value=True)
    if use_auto_rate and auto_risk_free_rate > 0:
        config['risk_free_rate'] = auto_risk_free_rate
        st.sidebar.success(f"Taux auto: {config['risk_free_rate']:.2%}")
    else:
        config['risk_free_rate'] = st.sidebar.number_input("Taux sans risque annuel (%)", min_value=0.0, max_value=20.0, value=auto_risk_free_rate * 100, step=0.1) / 100.0
        if not use_auto_rate:
            st.sidebar.info(f"Taux manuel: {config['risk_free_rate']:.2%}")

    return config

def render_performance_section(config, portfolio_value, total_invested, benchmark_value, portfolio_returns, benchmark_returns, benchmark_total_invested):
    """Affiche la section performance avec le graphique principal et les métriques."""
    st.header("Évolution et Performance")
    
    # --- Graphique d'évolution ---
    st.subheader(f"Évolution du capital sur : **{config['selected_period']}**")
    if config['dca_enabled']:
        total_dca_added = total_invested.iloc[-1] - config['initial_capital']
        st.info(f"**DCA activé** : {total_dca_added:,.0f}$ ajoutés sur la période.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value, mode='lines', name='Portefeuille', line=dict(color='royalblue', width=2)))
    fig.add_trace(go.Scatter(x=benchmark_value.index, y=benchmark_value, mode='lines', name=f'Indice ({config["benchmark"]})', line=dict(color='grey', width=2, dash='dash')))
    if config['dca_enabled']:
        fig.add_trace(go.Scatter(x=total_invested.index, y=total_invested, mode='lines', name='Capital Total Investi', line=dict(color='orange', width=1, dash='dot')))
    fig.update_layout(title="Évolution comparative des investissements", xaxis_title="Date", yaxis_title="Valeur ($)", hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # --- Métriques ---
    st.subheader("Métriques de Performance")
    p_simple, p_annual, p_vol, p_sharpe, p_twr = calculations.calculate_metrics_with_dca(portfolio_returns, portfolio_value, total_invested, config['risk_free_rate'])
    
    if config['dca_enabled']:
        b_total = (benchmark_value.iloc[-1] / benchmark_total_invested.iloc[-1]) - 1 if len(benchmark_value) > 0 else 0
        b_twr = (1 + benchmark_returns).prod() - 1 if len(benchmark_returns) > 0 else 0
        b_annual = ((1 + b_twr) ** (252 / len(benchmark_returns))) - 1 if len(benchmark_returns) > 0 else 0
    else:
        b_total = (benchmark_value.iloc[-1] / config['initial_capital']) - 1 if len(benchmark_value) > 0 else 0
        b_annual = ((1 + b_total) ** (252 / len(benchmark_returns))) - 1 if len(benchmark_returns) > 0 else 0

    b_vol = benchmark_returns.std() * np.sqrt(252) if len(benchmark_returns) > 0 else 0
    b_sharpe = (b_annual - config['risk_free_rate']) / (b_vol + 1e-10)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Portefeuille")
        st.metric("Rendement Annualisé (TWR)", f"{p_annual:.2%}", help="Rendement annualisé basé sur le Time-Weighted Return, indépendant des apports.")
        st.metric("Volatilité", f"{p_vol:.2%}")
        st.metric("Ratio de Sharpe", f"{p_sharpe:.2f}")
    with col2:
        st.markdown(f"#### {config['benchmark']}")
        st.metric("Rendement Annualisé", f"{b_annual:.2%}")
        st.metric("Volatilité", f"{b_vol:.2%}")
        st.metric("Ratio de Sharpe", f"{b_sharpe:.2f}")

def render_risk_analysis_section(portfolio_value, benchmark_value, portfolio_returns):
    """Affiche la section d'analyse des risques (Drawdown, Horizon, Alpha/Beta)."""
    st.header("Analyse des Risques et du Comportement")
    
    # Sélecteur
    analysis_view = st.radio("Choisir l'analyse", ["Évolution du Drawdown", "Horizon de Placement"], horizontal=True, label_visibility="collapsed")
    
    if analysis_view == "Évolution du Drawdown":
        # --- Drawdown ---
        st.subheader("Évolution du Drawdown")
        portfolio_drawdown = calculations.calculate_drawdown_series(portfolio_value)
        benchmark_drawdown = calculations.calculate_drawdown_series(benchmark_value)
        if not portfolio_drawdown.empty:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=portfolio_drawdown.index, y=portfolio_drawdown * 100, mode='lines', name='Drawdown Portefeuille', line=dict(color='red', width=2), fill='tozeroy'))
            fig_dd.add_trace(go.Scatter(x=benchmark_drawdown.index, y=benchmark_drawdown * 100, mode='lines', name='Drawdown Benchmark', line=dict(color='grey', width=1, dash='dash'), fill='tozeroy'))
            fig_dd.update_layout(title="Évolution du Drawdown au fil du temps", yaxis_title="Drawdown (%)", hovermode='x unified')
            st.plotly_chart(fig_dd, use_container_width=True)

            max_dd = calculations.calculate_max_drawdown(portfolio_value)
            st.metric("Drawdown Maximal du Portefeuille", f"{max_dd:.2%}", delta_color="inverse")
    
    else: # Horizon de Placement
        st.subheader("Analyse de l'Horizon de Placement")
        with st.spinner("Calcul en cours..."):
            holding_analysis = calculations.calculate_holding_period_analysis(portfolio_returns)
            if holding_analysis:
                horizons = sorted(holding_analysis.keys())
                probabilities = [holding_analysis[h]['probability_positive'] for h in horizons]
                labels = [holding_analysis[h]['label'] for h in horizons]
                
                fig_horizon = go.Figure()
                fig_horizon.add_trace(go.Scatter(x=horizons, y=probabilities, mode='lines+markers', name='Probabilité de Gain (%)', line=dict(color='green', width=3)))
                fig_horizon.update_layout(title="Probabilité de Gain selon l'Horizon de Placement", xaxis_title="Horizon de Placement (années)", yaxis_title="Probabilité de Gain (%)", yaxis_range=[0,105])
                st.plotly_chart(fig_horizon, use_container_width=True)
                
                # Tableau détaillé
                details_data = [{'Horizon': holding_analysis[h]['label'], 'Probabilité Gain (%)': f"{holding_analysis[h]['probability_positive']:.1f}%", 'Rendement Moyen Ann.': f"{holding_analysis[h]['annualized_avg_return']:.1%}"} for h in horizons]
                st.dataframe(pd.DataFrame(details_data), use_container_width=True, hide_index=True)
            else:
                st.warning("Données insuffisantes pour l'analyse de l'horizon de placement (minimum 2 ans requis).")
    
    st.markdown("---")
    st.subheader("Comportement par rapport au Marché")
    advanced_metrics = calculations.calculate_advanced_metrics(portfolio_returns, portfolio_returns, risk_free_rate=0.0) # Placeholder
    col1, col2, col3 = st.columns(3)
    col1.metric("Ratio de Sortino", f"{advanced_metrics['sortino_ratio']:.2f}", help="Ratio de Sharpe modifié qui ne pénalise que la volatilité négative.")
    col2.metric("Bêta (β)", f"{advanced_metrics['beta']:.2f}", help="Sensibilité aux mouvements du marché.")
    col3.metric("Alpha (α)", f"{advanced_metrics['alpha']:.2%}", help="Rendement excédentaire non expliqué par le marché.")

def render_composition_section(valid_tickers):
    """Affiche la section de composition du portefeuille (géographique et sectorielle)."""
    st.header("Composition et Diversification")

    # Récupération des données (peut être lent, à optimiser si nécessaire)
    sector_data, geo_data = [], []
    with st.spinner("Récupération des détails des actifs..."):
        for ticker in valid_tickers:
            try:
                info = yf.Ticker(ticker).info
                weight = st.session_state.weights.get(ticker, 0)
                sector_data.append({'Sector': info.get('sector', 'Inconnu'), 'Weight': weight})
                geo_data.append({'Country': info.get('country', 'Inconnu'), 'Weight': weight})
            except Exception:
                pass # Ignorer les tickers qui posent problème

    if not sector_data:
        st.warning("Impossible de récupérer les données de composition.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Répartition Sectorielle")
        sector_df = pd.DataFrame(sector_data).groupby('Sector')['Weight'].sum().reset_index()
        fig_sector = px.pie(sector_df, names='Sector', values='Weight', title='Répartition par Secteurs')
        fig_sector.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_sector, use_container_width=True)

    with col2:
        st.subheader("Répartition Géographique")
        geo_df = pd.DataFrame(geo_data).groupby('Country')['Weight'].sum().reset_index()
        fig_geo = px.pie(geo_df, names='Country', values='Weight', title='Répartition par Pays')
        fig_geo.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_geo, use_container_width=True)
        
def render_ai_analysis_section(config, portfolio_value, total_invested, benchmark_value, portfolio_returns, benchmark_returns, benchmark_total_invested, valid_tickers):
    st.header("Analyse Détaillée par Intelligence Artificielle")
    
    if st.button("Générer l'Analyse IA", type="primary"):
        with st.spinner("L'IA analyse votre portefeuille... (cela peut prendre jusqu'à 30 secondes)"):
            # Préparation des données pour l'IA
            p_simple, p_annual, p_vol, p_sharpe, p_twr = calculations.calculate_metrics_with_dca(portfolio_returns, portfolio_value, total_invested, config['risk_free_rate'])
            p_drawdown = calculations.calculate_max_drawdown(portfolio_value)
            
            if config['dca_enabled']:
                b_total = (benchmark_value.iloc[-1] / benchmark_total_invested.iloc[-1]) - 1
            else:
                b_total = (benchmark_value.iloc[-1] / config['initial_capital']) - 1
            b_annual = ((1 + b_total) ** (252 / len(benchmark_returns))) - 1
            b_vol = benchmark_returns.std() * np.sqrt(252)
            b_sharpe = (b_annual - config['risk_free_rate']) / (b_vol + 1e-10)
            b_drawdown = calculations.calculate_max_drawdown(benchmark_value)

            metrics_for_ai = {
                'portfolio_simple': p_simple, 'portfolio_annual': p_annual, 'portfolio_vol': p_vol, 'portfolio_sharpe': p_sharpe, 'portfolio_drawdown': p_drawdown,
                'benchmark_total': b_total, 'benchmark_annual': b_annual, 'benchmark_vol': b_vol, 'benchmark_sharpe': b_sharpe, 'benchmark_drawdown': b_drawdown
            }
            advanced_metrics = calculations.calculate_advanced_metrics(portfolio_returns, benchmark_returns, config['risk_free_rate'])

            benchmark_names = {"^GSPC": "S&P 500", "^IXIC": "NASDAQ", "GC=F": "Or", "DX-Y.NYB": "Dollar Index"}
            additional_context = {
                "benchmark_name": benchmark_names.get(config['benchmark'], config['benchmark']),
                "dca_enabled": config['dca_enabled'], "dca_frequency": config.get('dca_frequency'), "dca_amount": config.get('dca_amount'),
                "initial_capital": config['initial_capital'], "total_invested": total_invested.iloc[-1] if not total_invested.empty else config['initial_capital'],
                "risk_free_rate": config['risk_free_rate'],
            }

            # Génération et affichage
            ai_analysis = ai_analyzer.generate_portfolio_analysis(
                portfolio_value, benchmark_value, metrics_for_ai, advanced_metrics,
                config['selected_period'], valid_tickers, st.session_state.weights, additional_context
            )
            st.markdown(ai_analysis)

def render_guides_section():
    """Affiche les guides et informations d'aide."""
    st.header("Guides et Informations")
    with st.expander("Comprendre les métriques avancées"):
        st.markdown("""
        - **Ratio de Sortino**: Mesure le rendement ajusté au risque de baisse. Plus élevé = meilleur.
        - **Bêta (β)**: Indique la volatilité de votre portefeuille par rapport au marché. 1 = suit le marché, >1 = plus volatil, <1 = moins volatil.
        - **Alpha (α)**: Représente la surperformance (ou sous-performance) par rapport au rendement attendu compte tenu du risque. Positif = surperformance.
        """)
    with st.expander("Guide : Sauvegarde et Chargement de Portefeuilles"):
        st.markdown("""
        1. **Sauvegarder** : Configurez votre portefeuille et cliquez sur "Sauvegarder" pour télécharger un fichier JSON.
        2. **Charger** : Utilisez "Charger un Portefeuille" pour restaurer une configuration sauvegardée.
        """)