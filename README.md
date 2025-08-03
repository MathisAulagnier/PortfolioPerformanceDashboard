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
    git clone [https://github.com/votre-nom-utilisateur/votre-repo.git](https://github.com/votre-nom-utilisateur/votre-repo.git)
    cd votre-repo
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

5.  **Lancez l'application**
    ```sh
    streamlit run app.py
    ```

---

## Utilisation

Une fois l'application lancée, utilisez la barre latérale pour :
1.  Ajouter des tickers d'actions à votre portefeuille.
2.  Définir la pondération de chaque action (le total doit être de 100%).
3.  Choisir la période de backtest, le capital initial et l'indice de référence.
4.  Configurer les paramètres de DCA si vous le souhaitez.
5.  Explorer les différents onglets pour analyser les résultats.

---

## Licence

Distribué sous la licence MIT. Voir le fichier `LICENSE` pour plus d'informations.