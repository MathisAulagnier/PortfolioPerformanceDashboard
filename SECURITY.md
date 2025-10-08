# Sécurité et gestion des secrets

- Ne validez jamais `.streamlit/secrets.toml` dans Git (déjà ignoré dans `.gitignore`).
- Si une clé a été exposée, révoquez-la immédiatement depuis le fournisseur (ex: OpenAI) et générez une nouvelle clé.
- Purgez l'historique Git si nécessaire (BFG Repo-Cleaner ou git filter-repo) pour supprimer les secrets exposés des commits historiques.
- En production, préférez les variables d'environnement ou des gestionnaires de secrets (Vault, AWS Secrets Manager, etc.).
- Évitez d'appeler des API externes côté client; passez par le serveur quand c'est possible.
