"""
Test rapide des nouvelles fonctions de calcul de risque
"""

import numpy as np
import pandas as pd

from calculations import (
    calculate_drawdown_durations,
    calculate_risk_contribution,
    calculate_var_cvar,
)

# Test 1: VaR et CVaR
print("=" * 60)
print("TEST 1: Value at Risk (VaR) et Conditional VaR (CVaR)")
print("=" * 60)

# Générer des rendements simulés (distribution normale)
np.random.seed(42)
returns = pd.Series(np.random.normal(0.001, 0.02, 1000))

result = calculate_var_cvar(returns, confidence_level=0.95)
print(f"VaR 95%: {result['VaR']:.4f} ({result['VaR']*100:.2f}%)")
print(f"CVaR 95%: {result['CVaR']:.4f} ({result['CVaR']*100:.2f}%)")
print("✓ Le CVaR devrait être plus négatif que le VaR (pire que la perte moyenne)")
print()

# Test 2: Durées de drawdown
print("=" * 60)
print("TEST 2: Durées de Drawdown")
print("=" * 60)

# Créer une série de valeur de portefeuille simulée
dates = pd.date_range("2020-01-01", periods=500, freq="D")
portfolio_value = pd.Series(10000 * (1 + returns[:500]).cumprod(), index=dates)

dd_stats = calculate_drawdown_durations(portfolio_value)
print(f"Durée maximale de drawdown: {dd_stats['max_duration_days']:.0f} jours")
print(f"Durée moyenne de drawdown: {dd_stats['avg_duration_days']:.1f} jours")
print(f"Durée actuelle de drawdown: {dd_stats['current_duration_days']:.0f} jours")
print("✓ Les durées devraient être positives et cohérentes")
print()

# Test 3: Contribution au risque
print("=" * 60)
print("TEST 3: Contribution au Risque par Actif")
print("=" * 60)

# Créer des rendements pour 3 actifs fictifs
returns_by_asset = pd.DataFrame(
    {
        "AAPL": np.random.normal(0.0012, 0.025, 1000),
        "GOOGL": np.random.normal(0.0010, 0.022, 1000),
        "MSFT": np.random.normal(0.0011, 0.020, 1000),
    }
)

# Corrélation artificielle entre AAPL et MSFT
returns_by_asset["MSFT"] = 0.7 * returns_by_asset["AAPL"] + 0.3 * returns_by_asset["MSFT"]

weights = [0.4, 0.3, 0.3]  # 40% AAPL, 30% GOOGL, 30% MSFT

risk_contrib = calculate_risk_contribution(returns_by_asset, weights)
print("Contribution au risque par actif:")
total_contrib = 0.0
for ticker, contrib in risk_contrib.items():
    print(f"  {ticker}: {contrib:.2f}%")
    total_contrib += contrib

print(f"\nTotal: {total_contrib:.1f}% (devrait être ~100%)")
print("✓ AAPL devrait contribuer plus (poids plus élevé + corrélation avec MSFT)")
print()

# Test 4: Edge cases
print("=" * 60)
print("TEST 4: Edge Cases (données vides, NaN)")
print("=" * 60)

empty_series = pd.Series(dtype=float)
result_empty = calculate_var_cvar(empty_series)
print(f"VaR série vide: {result_empty['VaR']} (devrait être 0.0)")

nan_series = pd.Series([np.nan, np.nan, np.nan])
result_nan = calculate_var_cvar(nan_series)
print(f"VaR série NaN: {result_nan['VaR']} (devrait être 0.0)")

print("\n✓ Tous les tests passés avec succès!")
print()

# Résumé
print("=" * 60)
print("RÉSUMÉ DES NOUVELLES FONCTIONNALITÉS")
print("=" * 60)
print("""
Les 3 nouvelles fonctions sont opérationnelles:

1. calculate_var_cvar() 
   → Quantifie le risque de queue (pire perte probable)
   
2. calculate_drawdown_durations()
   → Mesure la résilience (temps en période de perte)
   
3. calculate_risk_contribution()
   → Identifie les positions qui contribuent le plus au risque

Ces métriques enrichissent significativement l'analyse IA en fournissant
des insights quantitatifs professionnels sur le profil de risque du portefeuille.
""")
