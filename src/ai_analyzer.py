from __future__ import annotations

import json

import streamlit as st
import yfinance as yf

from calculations import calculate_drawdown_series


def generate_portfolio_analysis(
    portfolio_value,
    benchmark_value,
    metrics: dict,
    advanced_metrics: dict,
    period: str,
    tickers_list: list[str],
    weights: dict[str, float],
    additional_context: dict | None = None,
):
    try:
        from openai import OpenAI

        api_key = st.secrets.get("OPENAI_API_KEY")
        print(f"Clé API chargée : {api_key is not None}")
        client = OpenAI(api_key=api_key)

        assets_details = {}
        for ticker in tickers_list:
            try:
                info = yf.Ticker(ticker).info
                mc = info.get("marketCap", 0) or 0
                assets_details[ticker] = {
                    "poids": float(weights.get(ticker, 0)),
                    "secteur": str(info.get("sectorKey", "Inconnu")),
                    "industrie": str(info.get("industryKey", "Inconnu")),
                    "pays": str(info.get("country", "Inconnu")),
                    "capitalisation": int(mc),
                }
            except Exception:
                assets_details[ticker] = {
                    "poids": float(weights.get(ticker, 0)),
                    "secteur": "Inconnu",
                    "industrie": "Inconnu",
                    "pays": "Inconnu",
                    "capitalisation": 0,
                }

        secteurs: dict[str, float] = {}
        pays: dict[str, float] = {}
        for _, d in assets_details.items():
            # use float(str(...)) to make mypy happy if poids is typed as object
            poids_val = float(str(d.get("poids", 0)))
            secteurs[str(d["secteur"])] = secteurs.get(str(d["secteur"]), 0.0) + poids_val
            pays[str(d["pays"])] = pays.get(str(d["pays"]), 0.0) + poids_val

        drawdown_series = calculate_drawdown_series(portfolio_value)
        drawdown_stats = {}
        if not drawdown_series.empty:
            dd_moy = (
                drawdown_series[drawdown_series < 0].mean() if (drawdown_series < 0).any() else 0
            )
            drawdown_stats = {"drawdown_moyen": f"{dd_moy:.2%}"}

        analysis_data = {
            "periode": str(period),
            "actifs": assets_details,
            "metriques": {
                "ret_total": f"{metrics.get('portfolio_simple', 0):.2%}",
                "ret_ann": f"{metrics.get('portfolio_annual', 0):.2%}",
                "vol": f"{metrics.get('portfolio_vol', 0):.2%}",
                "sharpe": f"{metrics.get('portfolio_sharpe', 0):.2f}",
                "alpha": f"{advanced_metrics.get('alpha', 0):.2%}",
                "beta": f"{advanced_metrics.get('beta', 0):.2f}",
            },
            "diversification": {"secteurs": secteurs, "pays": pays},
            "drawdown": drawdown_stats,
            "contexte": additional_context or {},
        }

        prompt = (
            "Analyse experte du portefeuille suivant en français:\n"
            + json.dumps(analysis_data, ensure_ascii=False, indent=2)
            + "\nFournir: Performance, Risques, Diversification, Recommandations concrètes."
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Expert financier"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1200,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur IA: {e}"
