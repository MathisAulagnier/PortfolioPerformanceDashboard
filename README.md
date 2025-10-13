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

### Sécurité des secrets

- Le fichier `.streamlit/secrets.toml` est ignoré par Git. Renseignez-y vos clés API uniquement en local.
- Ne commitez jamais de secrets. Si un secret a été exposé, révoquez-le immédiatement et remplacez-le par un nouveau.

---

## Licence

Distribué sous la licence MIT. Voir le fichier `LICENSE` pour plus d'informations.

---

## Changements récents

Ce projet a récemment reçu plusieurs corrections et optimisations visant à améliorer la compatibilité avec les versions récentes de Streamlit/Plotly et à réduire les appels réseau et le temps de calcul lors de l'utilisation interactive.

Principales modifications (résumé en français) :

- Remplacement des arguments dépréciés de Streamlit/Plotly : `width="stretch"` a été supprimé et les appels utilisent désormais `use_container_width=True` pour les graphiques et les tableaux.
- Réduction des appels réseau à yfinance : ajout d'une fonction mise en cache `get_ticker_info` (dans `src/data_manager.py`) pour éviter d'appeler `yf.Ticker(...).info` à répétition.
- Caching des calculs lourds : plusieurs fonctions de `src/calculations.py` sont désormais décorées avec `@st.cache_data` afin de diminuer les recomputations sur les reruns interactifs.
- Validation des tickers optimisée : les tests de disponibilité des tickers utilisent désormais `get_data(...)` (caché) au lieu d'appels directs non-cachés à `yf.Ticker().history`.
- Gestion du thème : l'application applique le thème stocké en session au démarrage de façon sûre (sans forcer systématiquement le mode sombre), et le bouton de changement de thème met à jour la session.
- OpenAI : création d'un helper OpenAI mis en cache et récupération plus robuste de la clé API (secrets ou variable d'environnement).
- Dépendance optionnelle `pycountry` : ajoutée aux dépendances pour permettre la cartographie ISO (utilisation optionnelle, protégée en cas d'absence).

Pourquoi ces changements ?

- Eviter les warnings et futures erreurs liés à des arguments dépréciés.
- Améliorer la réactivité de l'interface Streamlit en réduisant les allers-retours réseau et en mémorisant les résultats coûteux.
- Rendre le code plus robuste en cas d'erreurs réseau et fournir une meilleure expérience utilisateur (moins de latence lors d'explorations interactives).

Si vous souhaitez que j'ajoute `pycountry` au fichier `requirements.txt` ou que je crée une petite note de mise à jour plus formelle (CHANGELOG.md), je peux le faire sur demande.