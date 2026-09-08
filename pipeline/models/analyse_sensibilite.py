# -*- coding: utf-8 -*-
"""
Analyse de sensibilite : l'ecart de relation n'etant pas identifiable en
conditions reelles, on en evalue l'impact par scenarios de degradation volontaire de la relation
apprise -- translation de seuil et attenuation des coefficients (probabilites tirees vers 0.5) --
et on observe a partir de quelle ampleur le classement provincial predit s'inverse par rapport
au classement REEL (disponible ici uniquement parce que le scenario est simule sur donnees
completes -- dans un vrai deploiement, ce classement de reference ne serait pas connu, d'ou
l'interet de la batterie de scenarios plutot que d'un seul chiffre).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.stats import kendalltau
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

DATA_DIR = Path(r"C:\Users\hp\Desktop\MVP\these\pipeline\data")
SEED = 42
CATEGORICAL = ['fdsm121', 'fdsm111', 'fdsm202', 'fdsm122', 'fdsm201', 'cl108a', 'cl108b']
CONTINUOUS = ['fdsm126', 'fdsm127', 'fdsm128', 'fdsm213', 'fdsm214', 'fdsm215', 'fdsm216', 'fdsm217', 'fdsm218']
FEATURES = CATEGORICAL + CONTINUOUS
TARGETS = ['fdsm123', 'fdsm124', 'fdsm125']
BEST_XGB = {'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.03}
PROVINCES = {2: 'Haut-Ogooué', 3: 'Moyen-Ogooué', 4: 'Ngounié', 5: 'Nyanga',
             6: 'Ogooué-Ivindo', 7: 'Ogooué-Lolo', 8: 'Ogooué-Maritime', 9: 'Woleu-Ntem'}

estuaire = pd.read_parquet(DATA_DIR / "estuaire.parquet")
autres = pd.read_parquet(DATA_DIR / "autres_provinces.parquet")

def make_preprocess():
    return ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
        ('num', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value=0)),
                           ('scale', StandardScaler())]), CONTINUOUS),
    ])

ATTENUATIONS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
TRANSLATIONS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]  # decalage du seuil de decision autour de 0.5

results_atten = []
results_transl = []

for target in TARGETS:
    print(f"\n{'='*60}\nCIBLE: {target}\n{'='*60}")
    sub_train = estuaire.dropna(subset=[target]).copy()
    X_train = sub_train[FEATURES]
    y_train = (sub_train[target] == 1).astype(int)
    pipe = Pipeline([('prep', make_preprocess()),
                      ('clf', XGBClassifier(random_state=SEED, eval_metric='logloss', n_jobs=-1, **BEST_XGB))])
    pipe.fit(X_train, y_train)

    truth_col = f"{target}_verite"
    sub_test = autres.dropna(subset=[truth_col]).copy()
    proba = pipe.predict_proba(sub_test[FEATURES])[:, 1]
    sub_test = sub_test.assign(proba=proba, y_true=(sub_test[truth_col] == 1).astype(int))

    # classement REEL de reference : rang de chaque province par taux reel (1 = le plus eligible)
    taux_reel = sub_test.groupby('province')['y_true'].mean()
    rang_reel = {p: r for r, p in enumerate(taux_reel.sort_values(ascending=False).index)}
    provinces_list = list(rang_reel.keys())
    n_provinces = len(provinces_list)
    n_paires = n_provinces * (n_provinces - 1) // 2
    print(f"Classement reel (du + au - eligible): {[PROVINCES[p] for p in taux_reel.sort_values(ascending=False).index]}")

    def compter_inversions(taux_predit_series):
        rang_predit = {p: r for r, p in enumerate(taux_predit_series.sort_values(ascending=False).index)}
        n_inv = 0
        for p1, p2 in combinations(provinces_list, 2):
            ordre_reel = rang_reel[p1] < rang_reel[p2]
            ordre_predit = rang_predit[p1] < rang_predit[p2]
            if ordre_reel != ordre_predit:
                n_inv += 1
        tau, _ = kendalltau([rang_reel[p] for p in provinces_list], [rang_predit[p] for p in provinces_list])
        return n_inv, tau

    # ---- A. attenuation des probabilites vers 0.5 ----
    for f in ATTENUATIONS:
        proba_deg = 0.5 + f * (sub_test['proba'] - 0.5)
        taux_predit = sub_test.assign(p=proba_deg).groupby('province')['p'].mean()
        n_inversions, tau = compter_inversions(taux_predit)
        results_atten.append({'cible': target, 'facteur_attenuation': f, 'tau_kendall': tau,
                               'n_inversions_paires': n_inversions, 'n_paires_total': n_paires})
        print(f"  attenuation={f:.1f}: tau_kendall={tau:+.3f} inversions={n_inversions}/{n_paires}")

    # ---- B. translation du seuil de decision (taux predit = part de predict()==1) ----
    for d in TRANSLATIONS:
        for signe in ([0] if d == 0 else [+1, -1]):
            seuil = min(max(0.5 + signe * d, 0.01), 0.99)
            pred = (sub_test['proba'] >= seuil).astype(int)
            taux_predit = sub_test.assign(p=pred).groupby('province')['p'].mean()
            n_inversions, tau = compter_inversions(taux_predit)
            results_transl.append({'cible': target, 'decalage_seuil': signe * d, 'seuil_effectif': seuil,
                                    'tau_kendall': tau, 'n_inversions_paires': n_inversions, 'n_paires_total': n_paires})
        print(f"  seuil translate +/-{d}: fait")

pd.DataFrame(results_atten).to_csv(DATA_DIR / "resultats_sensibilite_attenuation.csv", index=False)
pd.DataFrame(results_transl).to_csv(DATA_DIR / "resultats_sensibilite_translation.csv", index=False)
print("\nTerminé.")
