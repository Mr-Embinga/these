# -*- coding: utf-8 -*-
"""
Etend la repondération au niveau département : reprend la même
mécanique par province (classifieur de séparabilité, poids tronqués, ré-entraînement pondéré),
mais agrège les prédictions par département plutôt que par province seule -- pour la carte de
production (taux corrigé). Réutilise l'extraction avec département de niveau3_par_departement.py.
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
import sqlite3

DATA_DIR = Path(r"C:\Users\hp\Desktop\MVP\these\pipeline\data")
SEED = 42
CATEGORICAL = ['fdsm121', 'fdsm111', 'fdsm202', 'fdsm122', 'fdsm201', 'cl108a', 'cl108b']
CONTINUOUS = ['fdsm126', 'fdsm127', 'fdsm128', 'fdsm213', 'fdsm214', 'fdsm215', 'fdsm216', 'fdsm217', 'fdsm218']
FEATURES = CATEGORICAL + CONTINUOUS
CIBLES_CORRIGEES = ['fdsm123', 'fdsm124']  # fibre reste au taux naif, arbitrage deja acte
BEST_XGB = {'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.03}
WEIGHT_CAP = 10.0
PROVINCES = {2: 'Haut-Ogooué', 3: 'Moyen-Ogooué', 4: 'Ngounié', 5: 'Nyanga',
             6: 'Ogooué-Ivindo', 7: 'Ogooué-Lolo', 8: 'Ogooué-Maritime', 9: 'Woleu-Ntem'}

estuaire = pd.read_parquet(DATA_DIR / "estuaire.parquet")

# reutilise l'extraction avec departement (deja construite dans niveau3_par_departement.py)
DB_STRUCT = r"C:\Users\hp\Desktop\MVP\dist\data\source\rgpl2023d_ds13x_100.csdb"
DB_LOC = r"C:\Users\hp\Desktop\atelier\rgpl2023_carto_csdb\rgpl2023d_cl23x_20250910.csdb"
con = sqlite3.connect(DB_STRUCT)
cols_ds1xx = [c for c in FEATURES if c not in ('cl108a', 'cl108b')] + CIBLES_CORRIGEES
cols = ", ".join(f"d.{c}" for c in cols_ds1xx)
q = f"""
SELECT l.fdsmc01 AS province, l.fdsmc02 AS departement, l.fdsmc03 AS canton,
       d.fdsm112 AS village_quartier, {cols}
FROM ds1xx d JOIN "level-1" l ON d."level-1-id" = l."level-1-id"
WHERE l.fdsmc01 != 1
"""
autres = pd.read_sql_query(q, con)
con.close()
autres['key'] = (autres['province'].astype(str) + "_" + autres['departement'].astype(str) + "_" +
                  autres['canton'].astype(str) + "_" + autres['village_quartier'].astype(str))
con = sqlite3.connect(DB_LOC)
loc = pd.read_sql_query(
    'SELECT l.clid01 province, l.clid02 departement, l.clid03 canton, l.clid04 village, '
    'c.cl108a, c.cl108b FROM cl1xx c JOIN "level-1" l ON c."level-1-id"=l."level-1-id"', con)
con.close()
loc['key'] = (loc['province'].astype(str) + "_" + loc['departement'].astype(str) + "_" +
              loc['canton'].astype(str) + "_" + loc['village'].astype(str))
autres = autres.merge(loc[['key', 'cl108a', 'cl108b']], on='key', how='left')
autres['cl108a'] = autres['cl108a'].fillna(0)
autres['cl108b'] = autres['cl108b'].fillna(0)
for c in FEATURES:
    autres[c] = pd.to_numeric(autres[c], errors='coerce')
autres['dept_code'] = (autres['province'].astype(int).astype(str).str.zfill(2) +
                        autres['departement'].astype(int).astype(str).str.zfill(2))

def make_preprocess():
    return ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
        ('num', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value=0)),
                           ('scale', StandardScaler())]), CONTINUOUS),
    ])

results = []
for pcode, pname in PROVINCES.items():
    print(f"\n{'='*50}\nPROVINCE: {pname}\n{'='*50}")
    prov_data = autres[autres['province'] == pcode]
    X_est = estuaire[FEATURES].copy()
    X_prov = prov_data[FEATURES].copy()

    X_dom = pd.concat([X_est, X_prov], ignore_index=True)
    y_dom = np.concatenate([np.zeros(len(X_est)), np.ones(len(X_prov))])
    dom_clf = Pipeline([('prep', make_preprocess()),
                         ('clf', RandomForestClassifier(n_estimators=150, max_depth=8, random_state=SEED, n_jobs=-1))])
    dom_clf.fit(X_dom, y_dom)
    p_prov = np.clip(dom_clf.predict_proba(X_est)[:, 1], 1e-3, 1 - 1e-3)
    weights_raw = p_prov / (1 - p_prov)
    weights = np.minimum(np.clip(weights_raw, weights_raw.min(), np.percentile(weights_raw, 99)), WEIGHT_CAP)

    for target in CIBLES_CORRIGEES:
        mask_train = estuaire[target].notna()
        pipe = Pipeline([('prep', make_preprocess()),
                          ('clf', XGBClassifier(random_state=SEED, eval_metric='logloss', n_jobs=-1, **BEST_XGB))])
        pipe.fit(X_est[mask_train.values], (estuaire.loc[mask_train, target] == 1).astype(int), clf__sample_weight=weights[mask_train.values])

        mask_test = prov_data[target].notna()
        proba = pipe.predict_proba(X_prov[mask_test.values])[:, 1]
        sub = prov_data.loc[mask_test].assign(proba=proba, y_true=(prov_data.loc[mask_test, target] == 1).astype(int))
        agg = sub.groupby('dept_code').agg(n=('y_true', 'size'), taux_reel=('y_true', 'mean'),
                                            taux_corrige=('proba', 'mean')).reset_index()
        agg['cible'] = target
        results.append(agg)
        print(f"  {target}: {len(agg)} departements, taux corrige moyen={agg['taux_corrige'].mean():.3f}")
    pd.DataFrame(pd.concat(results, ignore_index=True)).to_csv(DATA_DIR / "resultats_repondération_par_departement.csv", index=False)

print(f"\nSauvegarde: {DATA_DIR / 'resultats_repondération_par_departement.csv'}")
