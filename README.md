# Analyse et Backtesting de Portefeuille d'Investissement

Ce projet est un tableau de bord interactif développé en Python avec la bibliothèque Streamlit. Il a été conçu pour offrir un outil complet et intuitif permettant de réaliser des backtests de portefeuilles d'actions, d'analyser leurs performances et d'évaluer leurs risques face à des indices de référence.

L'objectif était de créer une application capable de simuler des stratégies d'investissement complexes, comme l'investissement programmé (DCA), et de fournir des métriques financières avancées pour une prise de décision éclairée.

---

## Fonctionnalités Principales

* **Configuration de Portefeuille Dynamique** : Ajoutez n'importe quel ticker, ajustez sa pondération et sauvegardez/chargez vos configurations.
* **Backtesting sur Périodes Variables** : Testez votre stratégie sur des horizons allant de 1 an à plus de 20 ans.
* **Simulation de l'Investissement Programmé (DCA)** : Activez le DCA pour simuler des apports périodiques et observez leur impact sur la performance.
* **Métriques de Performance Détaillées** : Accédez à des indicateurs clés comme le Rendement Annualisé, la Volatilité, le Ratio de Sharpe et le Time-Weighted Return (TWR) pour une analyse juste de la performance.
* **Analyse de Risque Avancée** : Évaluez le comportement de votre portefeuille avec des métriques comme l'Alpha, le Bêta, le Ratio de Sortino et une visualisation détaillée des Drawdowns.
* **🆕 Métriques de Risque Professionnelles** : VaR/CVaR pour quantifier le risque de queue, durées de drawdown, et contribution au risque par actif.
* **🆕 Analyse IA Enrichie** : Recommandations chiffrées et actionnables basées sur des métriques quantitatives avancées (powered by OpenAI).
* **Horizon de Placement** : Analysez la probabilité de gain de votre stratégie en fonction de la durée de détention.
* **Analyse de la Composition** : Visualisez la répartition géographique, sectorielle et industrielle de votre portefeuille.

---

## Aperçu de l'Interface

#### **Vue d'Ensemble des Performances**
Visualisez l'évolution de votre capital par rapport à un indice de référence et au capital total investi, particulièrement utile en cas de stratégie DCA.

![Vue d'ensemble](images/Screenshot%202025-08-03%20at%2017.29.30.png)

#### **Analyse des Risques**
Plongez dans l'analyse des pertes avec le graphique de Drawdown, qui montre les baisses depuis les sommets historiques et le temps de récupération.

![Analyse des Risques](images/Screenshot%202025-08-03%20at%2017.29.46.png)

#### **Horizon de Placement et Probabilité de Gain**
Déterminez la durée de détention nécessaire pour atteindre une probabilité de gain élevée, un outil puissant pour aligner votre stratégie avec vos objectifs à long terme.

![Horizon de Placement](images/Screenshot%202025-08-03%20at%2017.30.15.png)

#### **Tableaux de Métriques**
Comparez d'un seul coup d'œil les performances de votre portefeuille à celles de l'indice de référence grâce à des tableaux clairs et détaillés.

![Métriques de Performance](images/Screenshot%202025-08-03%20at%2017.30.25.png)

![Métriques Avancées](images/Screenshot%202025-08-03%20at%2017.30.35.png)

#### **Répartition du Portefeuille**
Comprenez la diversification de vos actifs grâce à une analyse visuelle de la répartition géographique et sectorielle.

![Répartition du Portefeuille](images/Screenshot%202025-08-03%20at%2017.31.07.png)

---

## Technologies Utilisées

* **Python** : Langage de programmation principal.
* **Streamlit** : Pour la création de l'interface web interactive.
* **Pandas** & **NumPy** : Pour la manipulation et l'analyse des données.
* **yfinance** : Pour la récupération des données de marché historiques.
* **Plotly** : Pour la génération des graphiques interactifs.
* **OpenAI API** : Pour la fonctionnalité optionnelle d'analyse par IA.
* **🆕 SciPy** : Pour les calculs statistiques avancés (VaR/CVaR).

---

## 🆕 Nouveautés - Analyse IA Professionnelle

### Métriques de Risque Avancées

L'application intègre désormais des métriques institutionnelles pour une analyse de niveau professionnel :

* **Value at Risk (VaR)** : Quantifie la perte maximale probable à 95% de confiance
* **Conditional VaR (CVaR)** : Mesure la moyenne des pertes au-delà du VaR (risque de queue)
* **Durées de Drawdown** : Analyse combien de temps le portefeuille reste en perte (max, moyenne, actuelle)
* **Contribution au Risque** : Identifie quels actifs contribuent le plus au risque total (prend en compte les corrélations)
* **Calmar Ratio** : Ratio rendement/drawdown maximum

### Analyse IA Enrichie

L'intelligence artificielle reçoit maintenant **3× plus de métriques quantitatives** pour générer des recommandations :

* ✅ **Analyses chiffrées** : Chaque recommandation est justifiée par des données précises
* ✅ **Recommendations actionnables** : Actions concrètes (ex: "Réduire AAPL de 55% → 40%")
* ✅ **Impact estimé** : Prévision de l'impact de chaque changement
* ✅ **Transparence totale** : Expander pour voir le prompt et les données envoyées à l'IA

### Guide Rapide

📚 **Documentation détaillée** :
- [`QUICK_START_GUIDE.md`](QUICK_START_GUIDE.md) - Guide de démarrage rapide
- [`IMPLEMENTATION_SUMMARY.md`](IMPLEMENTATION_SUMMARY.md) - Résumé complet des améliorations
- [`CHANGELOG_IA_IMPROVEMENTS.md`](CHANGELOG_IA_IMPROVEMENTS.md) - Détails techniques

🧪 **Tester les nouvelles fonctionnalités** :
```bash
python test_new_metrics.py  # Tests automatiques des métriques
```

---

## Démarrage

Suivez ces étapes pour lancer le projet sur votre machine locale.

### Prérequis

* Python 3.8 ou supérieur
* Un gestionnaire de paquets comme `pip`

### Installation

1.  **Clonez le dépôt**
    ```sh
    git clone https://github.com/MathisAulagnier/PortfolioPerformanceDashboard.git
    cd PortfolioPerformanceDashboard
    ```

2.  **Créez un environnement virtuel** (recommandé)
    ```sh
    python -m venv venv
    source venv/bin/activate  # Sur Windows: venv\Scripts\activate
    ```

3.  **Installez les dépendances**
    Assurez-vous d'avoir un fichier `requirements.txt` avec toutes les bibliothèques nécessaires.
    ```sh
    pip install -r requirements.txt
    ```

4.  **Configurez votre clé API (Optionnel)**
    Si vous souhaitez utiliser l'analyse par IA, créez un fichier `secrets.toml` dans un dossier `.streamlit` à la racine de votre projet.
    ```
    .
    ├── .streamlit/
    │   └── secrets.toml
    └── app.py
    ```
    Ajoutez votre clé API OpenAI dans le fichier `secrets.toml` :
    ```toml
    OPENAI_API_KEY = "votre_cle_api_ici"
    ```

5.  **Installez les dépendances et lancez l'application**
    ```sh
    pip install -r requirements.txt
    streamlit run src/main.py
    ```

### Option: Docker

Vous pouvez lancer l'application dans un conteneur:

```sh
docker build -t portfolio-dashboard .
docker run --rm -p 8501:8501 -e APP_PATH=src/b.py portfolio-dashboard
```

Ouvrez http://localhost:8501

---

## Utilisation

Une fois l'application lancée, utilisez la barre latérale pour :
1.  Ajouter des tickers d'actions à votre portefeuille.
2.  Définir la pondération de chaque action (le total doit être de 100%).
3.  Choisir la période de backtest, le capital initial et l'indice de référence.
4.  Configurer les paramètres de DCA si vous le souhaitez.
5.  Explorer les différents onglets pour analyser les résultats.

### Changement de thème

L'application démarre avec un **thème sombre** par défaut pour plus de confort visuel. Pour changer de thème :

**Via le menu Streamlit** (Recommandé) :
1. Cliquez sur le menu "⋮" (trois points) en haut à droite de l'application
2. Sélectionnez "Settings"
3. Dans la section "Theme", choisissez entre :
   - **Dark** : Thème sombre (par défaut)
   - **Light** : Thème clair
   - **Use system setting** : Utilise le réglage de votre système

**Personnalisation avancée** :
Pour personnaliser les couleurs du thème, modifiez le fichier `.streamlit/config.toml` :
```toml
[theme]
base = "dark"              # "light" ou "dark"
primaryColor = "#c98bdb"   # Couleur principale (violet)
backgroundColor = "#0E1117" # Couleur de fond
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"      # Couleur du texte
```

### Sécurité des secrets

- Le fichier `.streamlit/secrets.toml` est ignoré par Git. Renseignez-y vos clés API uniquement en local.
- Ne commitez jamais de secrets. Si un secret a été exposé, révoquez-le immédiatement et remplacez-le par un nouveau.

---

## Licence

Distribué sous la licence MIT. Voir le fichier `LICENSE` pour plus d'informations.

---
