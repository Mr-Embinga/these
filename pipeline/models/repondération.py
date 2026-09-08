# -*- coding: utf-8 -*-
"""
Repondération et correction de l'agrégat (référentiel 1.5).
Pour chaque province cible : classifieur de séparabilité Estuaire vs province (sur les seules
caractéristiques, jamais les variables de service), poids d'importance tronqués, ré-entraînement
pondéré du modèle retenu (XGBoost), et comparaison de l'agrégat corrigé vs naïf vs vérité réelle.

Point méthodologique important : la vérité (fdsm12x_verite) n'est utilisée ICI que pour VALIDER
si la repondération réduit effectivement le biais -- jamais pour calibrer la repondération
elle-même (celle-ci ne s'appuie que sur les caractéristiques, disponibles partout, condition
nécessaire pour qu'elle soit utilisable dans un vrai scénario de collecte interrompue).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

DATA_DIR = Path(r"C:\Users\hp\Desktop\MVP\these\pipeline\data")
OUT = DATA_DIR / "resultats_repondération.csv"
SEED = 42

CATEGORICAL = ['fdsm121', 'fdsm111', 'fdsm202', 'fdsm122', 'fdsm201', 'cl108a', 'cl108b']
CONTINUOUS = ['fdsm126', 'fdsm127', 'fdsm128', 'fdsm213', 'fdsm214', 'fdsm215', 'fdsm216', 'fdsm217', 'fdsm218']
FEATURES = CATEGORICAL + CONTINUOUS
TARGETS = ['fdsm123', 'fdsm124', 'fdsm125']
PROVINCES = {2: 'Haut-Ogooué', 3: 'Moyen-Ogooué', 4: 'Ngounié', 5: 'Nyanga',
             6: 'Ogooué-Ivindo', 7: 'Ogooué-Lolo', 8: 'Ogooué-Maritime', 9: 'Woleu-Ntem'}
BEST_XGB = {'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.03}
WEIGHT_CAP = 10.0  # troncature des poids extremes

estuaire = pd.read_parquet(DATA_DIR / "estuaire.parquet")
autres = pd.read_parquet(DATA_DIR / "autres_provinces.parquet")

def make_preprocess():
    return ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
        ('num', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value=0)),
                           ('scale', StandardScaler())]), CONTINUOUS),
    ])

results = []

for pcode, pname in PROVINCES.items():
    print(f"\n{'='*60}\nPROVINCE: {pname}\n{'='*60}")
    prov_data = autres[autres['province'] == pcode]
    X_est = estuaire[FEATURES].copy()
    X_prov = prov_data[FEATURES].copy()

    # ---- classifieur de separabilite Estuaire (0) vs province cible (1), caracteristiques seules ----
    X_dom = pd.concat([X_est, X_prov], ignore_index=True)
    y_dom = np.concatenate([np.zeros(len(X_est)), np.ones(len(X_prov))])
    dom_clf = Pipeline([('prep', make_preprocess()),
                         ('clf', RandomForestClassifier(n_estimators=150, max_depth=8, random_state=SEED, n_jobs=-1))])
    dom_clf.fit(X_dom, y_dom)
    p_prov = dom_clf.predict_proba(X_est)[:, 1]  # P(structure Estuaire ressemble a la province cible)
    p_prov = np.clip(p_prov, 1e-3, 1 - 1e-3)
    weights_raw = p_prov / (1 - p_prov)  # poids d'importance (densite ratio)
    weights = np.clip(weights_raw, weights_raw.min(), np.percentile(weights_raw, 99))
    weights = np.minimum(weights, WEIGHT_CAP)
    n_eff = (weights.sum() ** 2) / (weights ** 2).sum()
    print(f"  Poids: min={weights.min():.3f} median={np.median(weights):.3f} max={weights.max():.3f}")
    print(f"  Taille effective apres ponderation: {n_eff:,.0f} / {len(weights):,} ({n_eff/len(weights):.1%})")

    for target in TARGETS:
        mask_train = estuaire[target].notna()
        Xt = X_est[mask_train.values]
        yt = (estuaire.loc[mask_train, target] == 1).astype(int)
        wt = weights[mask_train.values]

        pipe = Pipeline([('prep', make_preprocess()),
                          ('clf', XGBClassifier(random_state=SEED, eval_metric='logloss', n_jobs=-1, **BEST_XGB))])
        pipe.fit(Xt, yt, clf__sample_weight=wt)

        truth_col = f"{target}_verite"
        mask_test = prov_data[truth_col].notna()
        X_test = X_prov[mask_test.values]
        y_true = (prov_data.loc[mask_test, truth_col] == 1).astype(int)
        proba = pipe.predict_proba(X_test)[:, 1]

        taux_reel = y_true.mean()
        taux_pondere = proba.mean()

        results.append({
            'province': pname, 'cible': target, 'n_eff_pct': n_eff / len(weights),
            'taux_reel': taux_reel, 'taux_pondere': taux_pondere,
            'erreur_ponderee_pts': (taux_pondere - taux_reel) * 100,
        })
        print(f"  {target}: reel={taux_reel:.3f} pondere={taux_pondere:.3f} "
              f"erreur={(taux_pondere-taux_reel)*100:+.1f}pts")

    pd.DataFrame(results).to_csv(OUT, index=False)

print(f"\nSauvegarde: {OUT}")
