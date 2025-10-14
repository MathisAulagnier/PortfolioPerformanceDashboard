from __future__ import annotations

import json
import os

import streamlit as st

from calculations import (
    calculate_drawdown_durations,
    calculate_drawdown_series,
    calculate_risk_contribution,
    calculate_var_cvar,
)
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
    portfolio_returns=None,
    returns_by_asset=None,
):
    try:
        from openai import OpenAI

        @st.cache_resource(ttl=3600)
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
            # Tentative de lecture depuis le fichier API.key
            try:
                api_key_file = os.path.join(os.path.dirname(__file__), "..", "API.key")
                if os.path.exists(api_key_file):
                    with open(api_key_file, "r") as f:
                        content = f.read()
                        # Chercher la ligne OpenAI API key
                        for line in content.split("\n"):
                            if "OpenAI API key" in line and ":" in line:
                                api_key = line.split(":", 1)[1].strip()
                                break
            except Exception:
                pass
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

        # Calcul des métriques de drawdown avancées
        drawdown_series = calculate_drawdown_series(portfolio_value)
        drawdown_stats = {}
        if not drawdown_series.empty:
            dd_moy = (
                drawdown_series[drawdown_series < 0].mean() if (drawdown_series < 0).any() else 0
            )
            drawdown_stats = {"drawdown_moyen": f"{dd_moy:.2%}"}

            # Ajout des durées de drawdown
            dd_durations = calculate_drawdown_durations(portfolio_value)
            drawdown_stats.update(
                {
                    "max_duration_jours": str(int(dd_durations.get("max_duration_days", 0))),
                    "avg_duration_jours": str(int(dd_durations.get("avg_duration_days", 0))),
                    "current_duration_jours": str(
                        int(dd_durations.get("current_duration_days", 0))
                    ),
                }
            )

        # Calcul VaR et CVaR
        risk_metrics = {}
        if portfolio_returns is not None and not portfolio_returns.empty:
            var_cvar = calculate_var_cvar(portfolio_returns, confidence_level=0.95)
            risk_metrics = {
                "VaR_95": f"{var_cvar.get('VaR', 0):.2%}",
                "CVaR_95": f"{var_cvar.get('CVaR', 0):.2%}",
            }

            # Calcul Calmar ratio (rendement annualisé / max drawdown)
            max_dd = abs(metrics.get("portfolio_max_dd", 0.01))
            calmar = metrics.get("portfolio_annual", 0) / max_dd if max_dd > 0 else 0
            risk_metrics["calmar_ratio"] = f"{calmar:.2f}"

        # Contribution au risque par actif
        risk_contribution = {}
        if returns_by_asset is not None and not returns_by_asset.empty:
            weights_list = [weights.get(ticker, 0) / 100.0 for ticker in returns_by_asset.columns]
            risk_contrib = calculate_risk_contribution(returns_by_asset, weights_list)
            # Ne garder que les 3 plus grands contributeurs pour le prompt
            sorted_contrib = sorted(risk_contrib.items(), key=lambda x: x[1], reverse=True)
            risk_contribution = {k: f"{v:.1f}%" for k, v in sorted_contrib[:5]}

        analysis_data = {
            "periode": str(period),
            "actifs": assets_details,
            "metriques_performance": {
                "ret_total": f"{metrics.get('portfolio_simple', 0):.2%}",
                "ret_annualise": f"{metrics.get('portfolio_annual', 0):.2%}",
                "volatilite": f"{metrics.get('portfolio_vol', 0):.2%}",
                "sharpe": f"{metrics.get('portfolio_sharpe', 0):.2f}",
                "sortino": f"{advanced_metrics.get('sortino_ratio', 0):.2f}",
                "alpha": f"{advanced_metrics.get('alpha', 0):.2%}",
                "beta": f"{advanced_metrics.get('beta', 0):.2f}",
            },
            "metriques_risque": risk_metrics,
            "drawdown": drawdown_stats,
            "contribution_risque_top5": risk_contribution,
            "diversification": {"secteurs": secteurs, "pays": pays},
            "contexte": additional_context or {},
        }

        # Prompt amélioré pour des recommandations structurées
        system_prompt = """Tu es un analyste quantitatif senior spécialisé en gestion de portefeuille.
Ton rôle est de fournir une analyse professionnelle, chiffrée et actionnables.
Réponds UNIQUEMENT en français et structure ta réponse ainsi:

1. **SYNTHÈSE** (2-3 phrases max): Vue d'ensemble performance/risque
2. **POINTS CLÉS** (3 bullets max):
   - Performance: force ou faiblesse principale
   - Risque: exposition ou vulnérabilité principale  
   - Diversification: concentration ou équilibre

3. **RECOMMANDATIONS** (3 actions concrètes, priorisées):
   Pour chaque recommandation:
   - Action précise (ex: "Réduire AAPL de 5% vers des obligations")
   - Justification chiffrée (utilise les métriques fournies)
   - Impact attendu (ex: "Réduire contribution au risque de 15% à 10%")

4. **VIGILANCE**: 1-2 points de surveillance à court terme
"""

        user_prompt = f"""Analyse ce portefeuille professionnel:

{json.dumps(analysis_data, ensure_ascii=False, indent=2)}

Concentre-toi sur:
- VaR/CVaR pour quantifier le risque de queue
- Contribution au risque pour identifier les concentrations
- Durées de drawdown pour évaluer la résilience
- Alpha/Beta pour la performance ajustée au risque

Fournis des recommandations concrètes et chiffrées."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1500,
            temperature=0.6,
        )

        # Retourner la réponse ET les données du prompt pour débogage
        result = {
            "analysis": response.choices[0].message.content,
            "prompt_data": {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "analysis_data": analysis_data,
            },
        }
        return result
    except Exception as e:
        return {"analysis": f"Erreur IA: {e}", "prompt_data": None}
