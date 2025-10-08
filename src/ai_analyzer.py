from __future__ import annotations

import json
import os

import streamlit as st

from calculations import calculate_drawdown_series
from data_manager import get_ticker_info


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

        @st.cache_data(ttl=3600)
        def get_openai_client(key: str):
            return OpenAI(api_key=key)

        api_key = None
        try:
            api_key = st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
        except Exception:
            api_key = None
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OpenAI API key not set")
        client = get_openai_client(api_key)

        assets_details = {}
        for ticker in tickers_list:
            info = get_ticker_info(ticker)
            mc = info.get("marketCap", 0) or 0
            assets_details[ticker] = {
                "poids": float(weights.get(ticker, 0)),
                "secteur": str(info.get("sectorKey", "Inconnu")),
                "industrie": str(info.get("industryKey", "Inconnu")),
                "pays": str(info.get("country", "Inconnu")),
                "capitalisation": int(mc) if mc else 0,
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
