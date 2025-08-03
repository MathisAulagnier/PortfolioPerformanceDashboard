# ai_analyzer.py
import streamlit as st
import yfinance as yf
import json
from datetime import datetime
from openai import OpenAI

# Import des fonctions des autres modules
from calculations import calculate_drawdown_series
from utils import convert_for_json

def generate_portfolio_analysis(portfolio_data, benchmark_data, metrics, advanced_metrics, period, tickers_list, weights, additional_context=None):
    """Génère une analyse détaillée du portefeuille via OpenAI GPT."""
    try:
        # NOTE: Assurez-vous que la clé API est bien dans les secrets de Streamlit
        # Correction: Utilisation de gpt-3.5-turbo comme dans le code original, et non gpt-4
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        assets_details = {}
        for ticker in tickers_list:
            try:
                info = yf.Ticker(ticker).info
                assets_details[ticker] = {
                    "poids": float(weights.get(ticker, 0)),
                    "secteur": str(info.get('sector', 'Inconnu')),
                    "pays": str(info.get('country', 'Inconnu')),
                    "nom_complet": str(info.get('longName', ticker))
                }
            except:
                assets_details[ticker] = {"poids": float(weights.get(ticker, 0)), "secteur": "Inconnu", "pays": "Inconnu", "nom_complet": str(ticker)}

        analysis_data = {
            "contexte_general": {"periode_analysee": str(period), "nombre_actifs": int(len(tickers_list))},
            "portefeuille": {
                "composition": assets_details,
                "metriques_performance": {k: f"{v:.2%}" if isinstance(v, (int, float)) else v for k, v in metrics.items() if 'portfolio' in k},
                "metriques_avancees": {k: f"{v:.2f}" if isinstance(v, (int, float)) else v for k, v in advanced_metrics.items()},
            },
            "benchmark": {
                "type": additional_context.get("benchmark_name", "Indice"),
                "metriques": {k: f"{v:.2%}" if isinstance(v, (int, float)) else v for k, v in metrics.items() if 'benchmark' in k},
            },
            "strategie_investissement": additional_context
        }
        
        analysis_data_json_safe = convert_for_json(analysis_data)
        
        prompt = f"""
En tant qu'analyste financier expert, analysez ce portefeuille. Soyez concis et pertinent.

DONNÉES DU PORTEFEUILLE:
{json.dumps(analysis_data_json_safe, indent=2, ensure_ascii=False)}

Fournissez une analyse structurée en 3 points :
1. **Performance et Risque**: Évaluez la performance (rendement, Sharpe) par rapport au benchmark et analysez le risque (volatilité, drawdown).
2. **Composition et Diversification**: Commentez la répartition sectorielle et géographique. Est-elle concentrée ou bien diversifiée ?
3. **Recommandations Stratégiques**: Donnez 2 à 3 conseils clairs et actionnables pour améliorer le portefeuille (ex: ajuster l'allocation, gérer un risque spécifique).
"""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Vous êtes un analyste financier expert fournissant des analyses claires et concises."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.5
        )
        return response.choices[0].message.content
        
    except Exception as e:
        # Affiche une erreur plus claire pour le débuggage
        st.error(f"Erreur de l'API OpenAI : {e}")
        return "L'analyse par IA n'a pas pu être générée. Veuillez vérifier votre clé API dans les secrets de Streamlit et réessayer."