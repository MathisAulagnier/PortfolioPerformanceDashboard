# data_manager.py
import streamlit as st
import yfinance as yf
import pandas as pd

@st.cache_data
def get_data(tickers, start, end):
    """Télécharge les données 'Close' pour une liste de tickers."""
    try:
        data = yf.download(tickers, start=start, end=end, progress=False)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        return data.dropna(how='all')
    except Exception as e:
        st.error(f"Erreur de téléchargement des données : {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_risk_free_rate():
    """Récupère le taux sans risque (Bons du Trésor US 13 semaines)."""
    try:
        treasury_data = yf.download("^IRX", period="5d", progress=False)
        if not treasury_data.empty and 'Close' in treasury_data.columns:
            return float(treasury_data['Close'].iloc[-1] / 100.0)
    except Exception:
        return 0.0
    return 0.0