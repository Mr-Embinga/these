# -*- coding: utf-8 -*-
"""
Application web (référentiel 1.6) -- intégration de l'algorithme d'apprentissage supervisé
(XGBoost, référentiel 1.4) dans une interface de consultation, traitement par lot et synthèse
provinciale, pour le Bureau Central du Recensement.

Séparation front/back (référentiel 1.6, contraintes) :
  - Partie « front » (visuelle) : templates Jinja2 (these/app/templates/), CSS (these/app/static/)
  - Partie « back » (fonctionnelle) : ce fichier -- chargement des modèles, prédiction,
    application du seuil de restitution, indicateur hors support.
"""
import io
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from functools import wraps
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request, send_file, flash, redirect, url_for, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "pipeline" / "models"
DATA_DIR = BASE_DIR.parent / "pipeline" / "data"

app = Flask(__name__)
# Cle de session : lue depuis une variable d'environnement en deploiement (voir render.yaml),
# repli sur une valeur de developpement local fixe pour ne pas invalider les sessions a chaque
# redemarrage pendant les tests.
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-cle-non-destinee-a-la-production")

# ---- securisation (referentiel : "liste des mesures mises en place avec capture d'ecran a
# l'appui") ----
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,   # cookie de session inaccessible en JavaScript (protection XSS)
    SESSION_COOKIE_SAMESITE="Lax",  # cookie non envoye depuis un site tiers (protection CSRF basique)
    # Secure force le cookie a n'etre envoye qu'en HTTPS -- desactive par defaut car le serveur de
    # demonstration tourne aussi en HTTP local ; a activer en deploiement reel (voir README).
    SESSION_COOKIE_SECURE=os.environ.get("FORCER_COOKIE_SECURE", "0") == "1",
)

JOURNAL = logging.getLogger("admin")
JOURNAL.setLevel(logging.INFO)
JOURNAL.propagate = False  # n'herite pas des handlers du logger racine (evite de melanger les
                            # lignes d'acces HTTP de werkzeug dans ce journal dedie)
_gestionnaire_journal = logging.FileHandler(BASE_DIR / "journal_admin.log", encoding="utf-8")
_gestionnaire_journal.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
JOURNAL.addHandler(_gestionnaire_journal)

# ---- back-office (referentiel : "acces administrateur au back office" exige dans le README du
# ZIP de depot) -- identifiants de test ci-dessous, a faire tourner via variables d'environnement
# avant tout deploiement reel (voir README).
ADMIN_UTILISATEUR = os.environ.get("ADMIN_UTILISATEUR", "admin")
ADMIN_MOTDEPASSE_HASH = os.environ.get(
    "ADMIN_MOTDEPASSE_HASH") or generate_password_hash("bcr-admin-2026")

# protection brute-force sur la connexion admin : au-dela de 5 echecs en 5 minutes pour une meme
# adresse IP, la connexion est bloquee temporairement (compteur en memoire, suffisant pour une
# demonstration mono-processus -- un deploiement multi-worker exigerait un stockage partage,
# type Redis).
TENTATIVES_ECHOUEES = defaultdict(list)
FENETRE_BLOCAGE_S = 300
MAX_TENTATIVES = 5

# ---- chargement des artefacts (une seule fois, au demarrage) ----
CONFIG = json.load(open(MODEL_DIR / "config.json", encoding="utf-8"))
LABELS = json.load(open(MODEL_DIR / "labels.json", encoding="utf-8"))
PROFILS_SUPPORT = set(json.load(open(MODEL_DIR / "profils_support.json", encoding="utf-8")))
AGREGATS = pd.read_csv(MODEL_DIR / "agregats_provinciaux.csv")

MODELES = {t: joblib.load(MODEL_DIR / f"modele_{t}.joblib") for t in CONFIG["targets"]}

# ---- donnees departement (cartes choroplethes) ----
NOMS_DEPARTEMENTS = {}
try:
    _geo = json.load(open(BASE_DIR / "static" / "departements.geojson", encoding="utf-8"))
    for _f in _geo["features"]:
        NOMS_DEPARTEMENTS[_f["properties"]["PolyCode"]] = _f["properties"]["PolyLabel"].title()
except FileNotFoundError:
    pass

NIVEAU3_DEPT = pd.read_csv(DATA_DIR / "resultats_niveau3_par_departement.csv",
                            dtype={"dept_code": str})
CIBLES_CORRIGEES = {"fdsm123", "fdsm124"}
try:
    PONDERE_DEPT = pd.read_csv(DATA_DIR / "resultats_repondération_par_departement.csv",
                                dtype={"dept_code": str})
except FileNotFoundError:
    PONDERE_DEPT = pd.DataFrame(columns=["dept_code", "n", "taux_reel", "taux_corrige", "cible"])

DECALAGE_PROVINCE = pd.read_csv(DATA_DIR / "decalage_distributions_par_province.csv")
NOM_VERS_CODE_PROVINCE = {v: k.zfill(2) for k, v in CONFIG["provinces"].items()}

CATEGORICAL = CONFIG["categorical"]
CONTINUOUS = CONFIG["continuous"]
FEATURES = CONFIG["features"]
TARGETS = CONFIG["targets"]
TARGET_LABELS = CONFIG["target_labels"]
SEUIL_RESTITUTION = CONFIG["seuil_restitution"]


SENTINEL_MANQUANT = -9  # doit rester identique a l'encodage utilise dans build_final_models.py


def encoder_profil(valeurs: dict) -> str:
    """Encodage canonique d'un profil catégoriel -- DOIT rester strictement identique à celui
    utilisé pour construire profils_support.json (build_final_models.py), sous peine de faire
    échouer silencieusement l'indicateur hors-support (suivi_problematiques_techniques.md,
    entrée 15)."""
    codes = []
    for c in CATEGORICAL:
        v = valeurs.get(c)
        codes.append(str(int(v)) if v is not None and not pd.isna(v) else str(SENTINEL_MANQUANT))
    return "|".join(codes)


def est_hors_support(valeurs_categorielles: dict) -> bool:
    """Indicateur référentiel 1.2 : le profil de la structure a-t-il été observé en Estuaire
    avec un effectif suffisant (>=30) ? Sinon, la prédiction extrapole hors du domaine appris."""
    return encoder_profil(valeurs_categorielles) not in PROFILS_SUPPORT


def predire_structure(donnees: dict) -> dict:
    """Prédit l'éligibilité aux trois réseaux pour une structure. `donnees` doit contenir
    toutes les colonnes FEATURES (valeurs numériques)."""
    X = pd.DataFrame([{c: donnees[c] for c in FEATURES}])
    resultats = {}
    for t in TARGETS:
        proba = float(MODELES[t].predict_proba(X)[:, 1][0])
        resultats[t] = {
            "reseau": TARGET_LABELS[t],
            "eligible": proba >= 0.5,
            "probabilite": round(proba, 3),
        }
    resultats["hors_support"] = est_hors_support(donnees)
    return resultats


@app.route("/")
def accueil():
    return render_template("index.html")


@app.route("/structure", methods=["GET", "POST"])
def structure():
    if request.method == "GET":
        return render_template("structure.html", categorical=CATEGORICAL, continuous=CONTINUOUS,
                                labels=LABELS)

    try:
        donnees = {}
        for c in CATEGORICAL:
            donnees[c] = float(request.form[c])
        for c in CONTINUOUS:
            valeur = request.form.get(c, "").strip()
            donnees[c] = float(valeur) if valeur else 0.0
    except (KeyError, ValueError):
        flash("Certains champs sont manquants ou invalides. Merci de vérifier la saisie.", "erreur")
        return redirect(url_for("structure"))

    resultats = predire_structure(donnees)
    return render_template("structure_resultat.html", resultats=resultats, donnees=donnees,
                            labels=LABELS, categorical=CATEGORICAL)


@app.route("/lot", methods=["GET", "POST"])
def lot():
    if request.method == "GET":
        return render_template("lot.html", features=FEATURES)

    fichier = request.files.get("fichier")
    if not fichier or fichier.filename == "":
        flash("Aucun fichier sélectionné.", "erreur")
        return redirect(url_for("lot"))

    try:
        if fichier.filename.endswith(".csv"):
            df = pd.read_csv(fichier)
        elif fichier.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(fichier)
        elif fichier.filename.endswith(".json"):
            df = pd.read_json(fichier)
        else:
            flash("Format non reconnu. Utilisez un fichier CSV, XLSX ou JSON.", "erreur")
            return redirect(url_for("lot"))
    except Exception:
        flash("Le fichier n'a pas pu être lu. Vérifiez son format.", "erreur")
        return redirect(url_for("lot"))

    manquantes = [c for c in FEATURES if c not in df.columns]
    if manquantes:
        flash(f"Colonnes manquantes dans le fichier : {', '.join(manquantes)}", "erreur")
        return redirect(url_for("lot"))

    df_pred = df.copy()
    for c in CONTINUOUS:
        df_pred[c] = pd.to_numeric(df_pred[c], errors="coerce").fillna(0.0)
    for c in CATEGORICAL:
        df_pred[c] = pd.to_numeric(df_pred[c], errors="coerce")

    for t in TARGETS:
        proba = MODELES[t].predict_proba(df_pred[FEATURES])[:, 1]
        df[f"{TARGET_LABELS[t]}_eligible"] = proba >= 0.5
        df[f"{TARGET_LABELS[t]}_probabilite"] = proba.round(3)

    profils = df_pred[CATEGORICAL].apply(lambda row: encoder_profil(row.to_dict()), axis=1)
    df["hors_support"] = ~profils.isin(PROFILS_SUPPORT)

    buffer = io.BytesIO()
    df.to_csv(buffer, index=False, encoding="utf-8-sig")
    buffer.seek(0)
    return send_file(buffer, mimetype="text/csv", as_attachment=True,
                      download_name="predictions_eligibilite.csv")


@app.route("/provinces")
def provinces():
    """Vue provinciale (référentiel 1.6) : taux estimé par province et service (corrigé par
    repondération quand elle a montré un bénéfice net -- eau, électricité -- naïf sinon --
    fibre, cf. arbitrage documenté dans note_correction_estimations.md §4), marge d'incertitude,
    écart de composition -- avec application du seuil de restitution (protection des données
    personnelles / secret statistique)."""
    data = AGREGATS.copy()
    data["type_estimation"] = data["correction_appliquee"].map(
        {True: "Corrigé (repondération)", False: "Naïf"})

    # tableau pivote : une ligne par province, l'ecart de composition (une seule valeur par
    # province, commune aux 3 reseaux) n'est plus repete sur 3 lignes -- evite l'ambiguite
    # "un JSD par (province, reseau) ?" alors qu'il n'y en a qu'un par province.
    provinces_triees = (data.drop_duplicates("province")
                         .sort_values("ecart_composition_jsd")["province"].tolist())
    tableau = []
    for province in provinces_triees:
        ligne = {"province": province}
        sous = data[data["province"] == province]
        ligne["ecart_composition"] = float(sous["ecart_composition_jsd"].iloc[0])
        for reseau in TARGET_LABELS.values():
            r = sous[sous["reseau"] == reseau].iloc[0]
            if r["restituable"]:
                ligne[reseau] = {"taux": f"{r['taux_retenu']*100:.1f} %",
                                  "marge": f"± {r['marge_incertitude_pts']:.1f} pts",
                                  "type": r["type_estimation"]}
            else:
                ligne[reseau] = {"taux": "Non restituable", "marge": "—", "type": "—"}
        tableau.append(ligne)

    return render_template("provinces.html", tableau=tableau, seuil=SEUIL_RESTITUTION,
                            reseaux=list(TARGET_LABELS.values()))


@app.route("/comparaison")
def comparaison():
    """Tableau de synthèse comparant les provinces entre elles, référentiel 1.6."""
    pivot = AGREGATS.pivot_table(index="province", columns="reseau", values="taux_retenu")
    ecart = AGREGATS.drop_duplicates("province").set_index("province")["ecart_composition_jsd"]
    pivot = pivot.join(ecart).sort_values("ecart_composition_jsd")
    lignes = []
    for province, row in pivot.iterrows():
        eau = float(row["eau"]) if pd.notna(row.get("eau")) else None
        elec = float(row["électricité"]) if pd.notna(row.get("électricité")) else None
        fibre = float(row["fibre"]) if pd.notna(row.get("fibre")) else None
        lignes.append({
            "province": province,
            "eau_pct": round(eau * 100, 1) if eau is not None else None,
            "electricite_pct": round(elec * 100, 1) if elec is not None else None,
            "fibre_pct": round(fibre * 100, 1) if fibre is not None else None,
            "ecart_composition": f"{row['ecart_composition_jsd']:.3f}",
        })

    taux_dept = []
    for cible in TARGETS:
        for dept_code, v in donnees_taux_departement(cible).items():
            taux_dept.append({"nom": v["nom"], "reseau": TARGET_LABELS[cible],
                               "taux": v["valeur"], "n": v["n"],
                               "type": "Corrigé" if v["corrige"] else "Naïf"})
    taux_dept.sort(key=lambda r: (r["reseau"], r["nom"]))

    return render_template("comparaison.html", lignes=lignes, taux_dept=taux_dept)


def donnees_taux_departement(cible):
    """Taux par département : corrigé (eau/électricité, repondération) ou naïf (fibre), même
    arbitrage qu'au niveau province (note_correction_estimations.md §4)."""
    if cible in CIBLES_CORRIGEES and not PONDERE_DEPT.empty:
        sub = PONDERE_DEPT[PONDERE_DEPT["cible"] == cible]
        return {r["dept_code"]: {"valeur": round(float(r["taux_corrige"]) * 100, 1), "n": int(r["n"]),
                                  "nom": NOMS_DEPARTEMENTS.get(r["dept_code"], r["dept_code"]),
                                  "corrige": True}
                for _, r in sub.iterrows()}
    sub = NIVEAU3_DEPT[NIVEAU3_DEPT["cible"] == cible]
    return {r["dept_code"]: {"valeur": round(float(r["taux_predit"]) * 100, 1), "n": int(r["n"]),
                              "nom": NOMS_DEPARTEMENTS.get(r["dept_code"], r["dept_code"]),
                              "corrige": False}
            for _, r in sub.iterrows()}


@app.route("/api/carte/<type_carte>/<cible>")
def api_carte(type_carte, cible):
    """Donnees departementales pour les cartes choroplethes (JS, decoratif -- le tableau
    accessible reste la source de verite RGAA). type_carte: 'erreur' (validation, Methode) ou
    'taux' (production, Provinces/Comparaison)."""
    if cible not in TARGET_LABELS:
        return jsonify({}), 404
    if type_carte == "erreur":
        sub = NIVEAU3_DEPT[NIVEAU3_DEPT["cible"] == cible]
        resultat = {r["dept_code"]: {"valeur": round(float(r["erreur_pts"]), 1), "n": int(r["n"]),
                                      "nom": NOMS_DEPARTEMENTS.get(r["dept_code"], r["dept_code"])}
                    for _, r in sub.iterrows()}
    elif type_carte == "taux":
        resultat = donnees_taux_departement(cible)
    else:
        resultat = {}
    return jsonify(resultat)


@app.route("/api/carte/ecart")
def api_carte_ecart():
    """Ecart de composition (JSD) par province, référentiel 1.2 — une seule mesure, indépendante
    du réseau (calculée sur l'ensemble des variables), pas une valeur par réseau."""
    resultat = {}
    for _, r in DECALAGE_PROVINCE.iterrows():
        code = NOM_VERS_CODE_PROVINCE.get(r["province"])
        if code:
            resultat[code] = {"valeur": round(float(r["JSD_moyenne"]) * 100, 1), "n": int(r["n"]),
                               "nom": r["province"]}
    return jsonify(resultat)


@app.route("/methode")
def methode():
    """Page Méthode/Validation -- rend l'application autoportante : un lecteur qui n'aurait pas
    lu le dossier doit pouvoir comprendre, depuis l'application seule, ce qui a été fait, pourquoi,
    et avec quelle fiabilité mesurée."""
    synthese = json.load(open(MODEL_DIR / "synthese_methode.json", encoding="utf-8"))
    importances = pd.read_csv(MODEL_DIR / "importance_variables.csv")
    importances_par_reseau = {
        reseau: grp[["variable", "importance"]].to_dict(orient="records")
        for reseau, grp in importances.groupby("reseau")
    }
    erreur_dept = NIVEAU3_DEPT.copy()
    erreur_dept["nom"] = erreur_dept["dept_code"].map(NOMS_DEPARTEMENTS).fillna(erreur_dept["dept_code"])
    erreur_dept = erreur_dept.sort_values(["reseau", "nom"])
    return render_template("methode.html", synthese=synthese, importances=importances_par_reseau,
                            erreur_dept=erreur_dept.to_dict(orient="records"))


def admin_requis(vue):
    """Protège une route back-office : redirige vers la connexion si la session n'est pas
    authentifiée (référentiel -- séparation d'un espace administrateur du reste de l'application,
    accès non exposé publiquement)."""
    @wraps(vue)
    def wrapper(*args, **kwargs):
        if not session.get("admin_authentifie"):
            return redirect(url_for("admin_connexion", suivant=request.path))
        return vue(*args, **kwargs)
    return wrapper


@app.route("/admin/connexion", methods=["GET", "POST"])
def admin_connexion():
    if request.method == "GET":
        return render_template("admin_connexion.html")

    ip = request.remote_addr
    maintenant = time.time()
    TENTATIVES_ECHOUEES[ip] = [t for t in TENTATIVES_ECHOUEES[ip] if maintenant - t < FENETRE_BLOCAGE_S]
    if len(TENTATIVES_ECHOUEES[ip]) >= MAX_TENTATIVES:
        JOURNAL.info(f"connexion bloquee (trop de tentatives) ip={ip}")
        flash("Trop de tentatives échouées. Réessayez dans quelques minutes.", "erreur")
        return redirect(url_for("admin_connexion"))

    utilisateur = request.form.get("utilisateur", "")
    mot_de_passe = request.form.get("mot_de_passe", "")
    if utilisateur == ADMIN_UTILISATEUR and check_password_hash(ADMIN_MOTDEPASSE_HASH, mot_de_passe):
        session["admin_authentifie"] = True
        TENTATIVES_ECHOUEES[ip].clear()
        JOURNAL.info(f"connexion reussie ip={ip} utilisateur={utilisateur}")
        return redirect(request.args.get("suivant") or url_for("admin_tableau_de_bord"))

    TENTATIVES_ECHOUEES[ip].append(maintenant)
    JOURNAL.info(f"connexion echouee ip={ip} utilisateur={utilisateur!r}")
    flash("Identifiants incorrects.", "erreur")
    return redirect(url_for("admin_connexion"))


@app.route("/admin/deconnexion")
def admin_deconnexion():
    session.pop("admin_authentifie", None)
    JOURNAL.info(f"deconnexion ip={request.remote_addr}")
    return redirect(url_for("accueil"))


@app.route("/admin")
@admin_requis
def admin_tableau_de_bord():
    """Back-office : métadonnées du modèle retenu non exposées publiquement (hyperparamètres,
    tailles des jeux de données, date d'entraînement) et comparaison complète des six modèles
    testés (le public ne voit que le modèle retenu, en section Méthode & validation)."""
    niveau1 = pd.read_csv(DATA_DIR / "resultats_niveau1.csv")
    try:
        ftt = pd.read_csv(DATA_DIR / "resultats_ft_transformer.csv")
        niveau1 = pd.concat([niveau1, ftt], ignore_index=True)
    except FileNotFoundError:
        pass
    niveau1["cible_libelle"] = niveau1["cible"].map(TARGET_LABELS)

    hyperparametres = None
    ligne_xgb = niveau1[(niveau1["modele"] == "gradient_boosting_xgb") & niveau1["meilleurs_params"].notna()]
    if not ligne_xgb.empty:
        hyperparametres = ligne_xgb.iloc[0]["meilleurs_params"]

    fichiers_modeles = [MODEL_DIR / f"modele_{t}.joblib" for t in TARGETS]
    dates = [datetime.fromtimestamp(p.stat().st_mtime) for p in fichiers_modeles if p.exists()]
    date_entrainement = max(dates) if dates else None

    # Effectifs fixes plutot que lus depuis les fichiers sources (estuaire.parquet /
    # autres_provinces.parquet) : ces fichiers portent les caracteristiques individuelles de
    # chaque structure et n'ont pas besoin d'etre presents sur le serveur deploye pour ce simple
    # affichage de deux totaux, deja verifies et stables (vii.1 du dossier).
    tailles = {"estuaire": 502940, "autres_provinces": 390376}

    comparaison = niveau1.sort_values(["cible_libelle", "modele"]).copy()
    # temps_s est une colonne float64 : assigner None dedans est silencieusement reconverti en
    # NaN par pandas (un float64 ne peut pas stocker Python None) sauf a la faire passer en
    # dtype object d'abord -- sinon le gabarit affiche "nan" au lieu de "-" pour FT-Transformer,
    # qui n'a pas de temps mesure (trouve par verification visuelle, pas par relecture du code).
    comparaison["temps_s"] = comparaison["temps_s"].astype(object)
    comparaison = comparaison.where(pd.notna(comparaison), None)

    return render_template(
        "admin_tableau_de_bord.html",
        config=CONFIG,
        hyperparametres=hyperparametres,
        date_entrainement=date_entrainement,
        tailles=tailles,
        comparaison=comparaison.to_dict(orient="records"),
        seuil=SEUIL_RESTITUTION,
    )


if __name__ == "__main__":
    # use_reloader=False : le rechargeur automatique (watchdog) declenche des faux positifs sur
    # cette machine Windows (detecte des "changements" dans les fichiers installes de Flask
    # lui-meme et redemarre en boucle, coupant les connexions en cours) -- voir
    # suivi_problematiques_techniques.md. Redemarrer manuellement apres modification du code.
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5000)
