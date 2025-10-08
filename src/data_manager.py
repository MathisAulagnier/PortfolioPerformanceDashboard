import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data
def get_data(tickers: list[str], start, end) -> pd.DataFrame:
    """Télécharge et met en cache les prix de clôture pour une liste de tickers.

    Retourne un DataFrame (colonnes = tickers, index = dates) sans lignes entièrement vides.
    """
    try:
        # Pass explicit auto_adjust to avoid FutureWarning when the default changes
        data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)[
            "Close"
        ]
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        return data.dropna(how="all")
    except Exception as e:
        st.error(f"Erreur lors du téléchargement des données : {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_risk_free_rate() -> float:
    """Récupère le taux sans risque (bons du Trésor US 13 semaines) en décimal."""
    try:
        # Pass explicit auto_adjust to avoid FutureWarning
        treasury_data = yf.download("^IRX", period="5d", progress=False, auto_adjust=False)
        if not treasury_data.empty and "Close" in treasury_data.columns:
            # Ensure we extract a Python float (avoid deprecated float(Series))
            latest_rate = float(treasury_data["Close"].iloc[-1]) / 100.0
            return latest_rate
        return 0.0
    except Exception:
        return 0.0


@st.cache_data(ttl=3600)
def get_ticker_info(ticker: str) -> dict:
    """Récupère et met en cache les informations d'un ticker via yfinance.

    Retourne un dictionnaire vide si une erreur survient.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        # Convertir en dict standard (au cas où yfinance retourne un objet spécial)
        return dict(info)
    except Exception:
        return {}
