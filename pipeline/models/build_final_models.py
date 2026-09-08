# -*- coding: utf-8 -*-
"""
Entraîne et sauvegarde sur disque les modèles de production (un par réseau), plus les artefacts
nécessaires à l'application : profils de support commun (référentiel 1.2, indicateur "hors
support"), agrégats provinciaux (taux prédit, marge d'incertitude, écart de composition).
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

DATA_DIR = Path(r"C:\Users\hp\Desktop\MVP\these\pipeline\data")
MODEL_DIR = Path(r"C:\Users\hp\Desktop\MVP\these\pipeline\models")
MODEL_DIR.mkdir(exist_ok=True)
SEED = 42

CATEGORICAL = ['fdsm121', 'fdsm111', 'fdsm202', 'fdsm122', 'fdsm201', 'cl108a', 'cl108b']
CONTINUOUS = ['fdsm126', 'fdsm127', 'fdsm128', 'fdsm213', 'fdsm214', 'fdsm215', 'fdsm216', 'fdsm217', 'fdsm218']
FEATURES = CATEGORICAL + CONTINUOUS
TARGETS = ['fdsm123', 'fdsm124', 'fdsm125']
BEST_XGB = {'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.03}
PROVINCES = {1: 'Estuaire', 2: 'Haut-Ogooué', 3: 'Moyen-Ogooué', 4: 'Ngounié', 5: 'Nyanga',
             6: 'Ogooué-Ivindo', 7: 'Ogooué-Lolo', 8: 'Ogooué-Maritime', 9: 'Woleu-Ntem'}
TARGET_LABELS = {'fdsm123': 'eau', 'fdsm124': 'électricité', 'fdsm125': 'fibre'}
SEUIL_RESTITUTION = 10  # taille minimale pour publier un taux (secret statistique)

estuaire = pd.read_parquet(DATA_DIR / "estuaire.parquet")
autres = pd.read_parquet(DATA_DIR / "autres_provinces.parquet")

def make_preprocess():
    return ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
        ('num', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value=0)),
                           ('scale', StandardScaler())]), CONTINUOUS),
    ])

# ---- 1. modeles de production, un par reseau, entraines sur 100% Estuaire ----
print("[1/3] Entraînement des modèles de production...")
for target in TARGETS:
    sub = estuaire.dropna(subset=[target])
    pipe = Pipeline([('prep', make_preprocess()),
                      ('clf', XGBClassifier(random_state=SEED, eval_metric='logloss', n_jobs=-1, **BEST_XGB))])
    pipe.fit(sub[FEATURES], (sub[target] == 1).astype(int))
    joblib.dump(pipe, MODEL_DIR / f"modele_{target}.joblib")
    print(f"  {target} ({TARGET_LABELS[target]}) -> sauvegardé")

# ---- 2. profils de support commun (Estuaire) pour l'indicateur "hors support" ----
print("[2/3] Construction des profils de support (référentiel 1.2)...")
# encodage entier explicite (pas astype(str) direct sur des colonnes float, qui produirait
# "1.0"/"nan" au lieu de "1" -- doit rester strictement identique a l'encodage cote application,
# voir suivi_problematiques_techniques.md, entree 14/15)
prof_cols = estuaire[CATEGORICAL].fillna(-9).astype(int).astype(str)
prof = prof_cols.agg('|'.join, axis=1)
profils_connus = set(prof[prof.map(prof.value_counts()) >= 30])  # profils avec >=30 observations
with open(MODEL_DIR / "profils_support.json", "w", encoding="utf-8") as f:
    json.dump(sorted(profils_connus), f)
print(f"  {len(profils_connus):,} profils distincts avec support suffisant (>=30 obs)")

# ---- 3. agregats provinciaux : taux predit (naif), marge d'incertitude, ecart de composition ----
print("[3/3] Agrégats provinciaux pour la vue de synthèse...")
# recharge les resultats deja produits (niveau3, incertitude, decalage) plutot que de tout recalculer
niveau3_prov = pd.read_csv(DATA_DIR / "resultats_niveau3_par_province.csv")
sousdomaines = pd.read_csv(DATA_DIR / "resultats_niveau2_sousdomaines.csv")
decalage = pd.read_csv(r"C:\Users\hp\Desktop\MVP\dist\data\decalage_distributions_par_province.csv")

marge_par_cible = sousdomaines.groupby('cible')['erreur_pts'].apply(lambda s: s.abs().max()).to_dict()

agregats = []
for target in TARGETS:
    for pcode, pname in PROVINCES.items():
        if pcode == 1:
            continue  # Estuaire = base d'apprentissage, pas une cible de prediction
        row = niveau3_prov[(niveau3_prov['cible'] == target) & (niveau3_prov['province'] == pname)]
        dec_row = decalage[decalage['province'] == pname]
        n = int(row['n'].iloc[0]) if len(row) else None
        agregats.append({
            'cible': target, 'reseau': TARGET_LABELS[target], 'province': pname,
            'n': n,
            'taux_predit_naif': float(row['taux_predit_naif'].iloc[0]) if len(row) else None,
            'marge_incertitude_pts': float(marge_par_cible.get(target, float('nan'))),
            'ecart_composition_jsd': float(dec_row['JSD_moyenne'].iloc[0]) if len(dec_row) else None,
            'restituable': (n or 0) >= SEUIL_RESTITUTION,
        })
pd.DataFrame(agregats).to_csv(MODEL_DIR / "agregats_provinciaux.csv", index=False)
print(f"  Sauvegardé: {MODEL_DIR / 'agregats_provinciaux.csv'}")

with open(MODEL_DIR / "config.json", "w", encoding="utf-8") as f:
    json.dump({'features': FEATURES, 'categorical': CATEGORICAL, 'continuous': CONTINUOUS,
               'targets': TARGETS, 'target_labels': TARGET_LABELS, 'provinces': PROVINCES,
               'seuil_restitution': SEUIL_RESTITUTION}, f, ensure_ascii=False, indent=2)
print("\nTerminé.")
