# Éligibilité des structures aux réseaux — application de démonstration

Application Flask du Bureau Central du Recensement (BCR) : prédiction de l'éligibilité des
structures aux réseaux d'eau, d'électricité et de fibre optique à partir du RGPL. Thèse
professionnelle, Mastère Chef de projet Data et Intelligence Artificielle (RNCP 37137).

## Prérequis

- Python 3.12 ou supérieur
- Les artefacts du pipeline déjà générés dans `these/pipeline/models/` et `these/pipeline/data/`
  (modèles `.joblib`, `config.json`, `labels.json`, `profils_support.json`,
  `agregats_provinciaux.csv` et fichiers `resultats_*.csv` — produits par les scripts de
  `these/pipeline/`, non régénérés automatiquement au lancement de l'application)

## Installation

```bash
cd these/app
pip install -r requirements.txt
python app.py
```

L'application démarre sur `http://127.0.0.1:5000`. Le rechargement automatique est désactivé
(`use_reloader=False` — voir `suivi_problematiques_techniques.md`, entrée 13) : après toute
modification de `app.py`, arrêter puis relancer le processus manuellement. Les templates, le CSS
et le JavaScript, eux, sont pris en compte au simple rafraîchissement du navigateur.

## Identifiants de test

**Consultation et traitement par lot** : aucune authentification requise (fonctionnalités
publiques).

**Back-office administrateur** (`/admin`) :
- Identifiant : `admin`
- Mot de passe : `bcr-admin-2026`

Ces identifiants sont définis par défaut dans `app.py` (hachage du mot de passe, jamais stocké en
clair) et peuvent être surchargés par variables d'environnement avant tout déploiement réel :

```bash
export ADMIN_UTILISATEUR="..."
export ADMIN_MOTDEPASSE_HASH="$(python -c 'from werkzeug.security import generate_password_hash; print(generate_password_hash(\"...\"))')"
export FORCER_COOKIE_SECURE=1   # a activer uniquement derriere HTTPS
```

Au-delà de 5 échecs de connexion en 5 minutes pour une même adresse IP, la connexion admin est
bloquée temporairement (protection brute-force, voir `note_securisation_et_accessibilite.md`).

## Base de données

L'application ne se connecte pas à un serveur de base de données au moment de l'exécution : elle
charge les modèles entraînés (`.joblib`) et des tables pré-agrégées (CSV) produites par le
pipeline, choix documenté (voir `these/sql/build_database.py`, décision d'architecture). Une base
SQL est fournie séparément comme artefact du référentiel (installation de la base et banc d'essai
indexé/non-indexé, voir `note_banc_essai_sql.md`) :

- `these/sql/rgpl_structures.db` — base SQLite autonome (fichier local, aucun serveur ni
  identifiant de connexion nécessaires — s'ouvre avec n'importe quel client SQLite, par exemple
  `sqlite3 these/sql/rgpl_structures.db`)
- `these/sql/rgpl_structures_dump.sql` — export SQL complet de cette base

## Compatibilité navigateur

Testée sous Chrome (vérification visuelle interactive : formulaires, back-office, blocage
brute-force) et sous Firefox et Edge (test automatisé via Playwright -- chargement des six pages
principales, capture des erreurs console, comptage des éléments SVG/tableaux rendus). Zéro
erreur console sur les
trois navigateurs après correction d'un favicon manquant (404 cosmétique, sans impact
fonctionnel). Captures d'écran comparatives Firefox/Edge des cartes et graphiques :
`these/securisation/capture_firefox_*.jpg`, `these/securisation/capture_edge_*.jpg` — rendu
visuellement identique, cohérent avec l'absence d'API spécifique à un navigateur dans le code
(HTML5, CSS standard, D3.js v7).

## Accessibilité

Audit outillé (axe-core 4.9.1) mené sur les 9 pages/états de l'application : zéro violation.
Détail dans `note_securisation_et_accessibilite.md`.

## Déploiement (Render, gratuit)

Le dépôt racine (`these/`, pas `these/app/`) contient un fichier `render.yaml` : Render le lit
automatiquement et configure le service sans manipulation dans le tableau de bord (build, commande
de démarrage `gunicorn`, variable `FLASK_SECRET_KEY` générée aléatoirement, cookies forcés en
HTTPS). Étapes : pousser ce dépôt sur GitHub, créer un compte Render, "New +" → "Blueprint",
connecter le dépôt. HTTPS est fourni automatiquement par Render sur le sous-domaine
`*.onrender.com`, sans configuration ni coût supplémentaire.

L'offre gratuite met le service en veille après ~15 minutes d'inactivité (30-60 secondes pour
redémarrer au prochain accès) — à garder en tête avant une démonstration en direct.

## Structure du code

- `app.py` — partie « back » : chargement des modèles, prédiction, seuil de restitution,
  indicateur hors support, back-office.
- `templates/` — partie « front » : gabarits Jinja2.
- `static/` — CSS, JavaScript (cartes D3, tableaux triables), bibliothèques.
