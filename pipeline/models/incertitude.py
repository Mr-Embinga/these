# -*- coding: utf-8 -*-
"""
Estimation de l'incertitude  :
  A. Calibration de la probabilité au niveau structure (interne Estuaire vs hors domaine)
  B. Marge par sous-domaine (niveau 2, leave-one-subdomain-out) -- dispersion d'erreur
  C. Incertitude irréductible -- taux conditionnel par profil catégoriel (plancher bayésien approché)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
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

estuaire = pd.read_parquet(DATA_DIR / "estuaire.parquet")
autres = pd.read_parquet(DATA_DIR / "autres_provinces.parquet")

def make_preprocess():
    return ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
        ('num', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value=0)),
                           ('scale', StandardScaler())]), CONTINUOUS),
    ])

def ece(y_true, proba, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(proba, bins[1:-1])
    rows = []
    total_err = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        mean_pred = proba[mask].mean()
        obs_rate = y_true[mask].mean()
        w = mask.sum() / len(proba)
        total_err += w * abs(mean_pred - obs_rate)
        rows.append({'bin': b, 'n': int(mask.sum()), 'proba_moyenne': mean_pred, 'taux_observe': obs_rate})
    return total_err, pd.DataFrame(rows)

# ============================================================
# A. CALIBRATION -- interne (Estuaire, split simple) vs hors domaine (autres provinces)
# ============================================================
print("="*60, "\nA. CALIBRATION\n", "="*60)
calib_results = []
for target in TARGETS:
    sub = estuaire.dropna(subset=[target]).copy()
    X, y = sub[FEATURES], (sub[target] == 1).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    pipe = Pipeline([('prep', make_preprocess()),
                      ('clf', XGBClassifier(random_state=SEED, eval_metric='logloss', n_jobs=-1, **BEST_XGB))])
    pipe.fit(X_tr, y_tr)
    proba_interne = pipe.predict_proba(X_te)[:, 1]
    ece_interne, _ = ece(y_te.values, proba_interne)

    truth_col = f"{target}_verite"
    sub_ext = autres.dropna(subset=[truth_col]).copy()
    proba_externe = pipe.predict_proba(sub_ext[FEATURES])[:, 1]
    y_ext = (sub_ext[truth_col] == 1).astype(int).values
    ece_externe, _ = ece(y_ext, proba_externe)

    calib_results.append({'cible': target, 'ece_interne_estuaire': ece_interne, 'ece_externe_autres_provinces': ece_externe})
    print(f"{target}: ECE interne={ece_interne:.4f}  ECE hors-domaine={ece_externe:.4f}  "
          f"(degradation x{ece_externe/max(ece_interne,1e-6):.1f})")

pd.DataFrame(calib_results).to_csv(DATA_DIR / "resultats_calibration.csv", index=False)

# ============================================================
# B. MARGE PAR SOUS-DOMAINE (niveau 2, leave-one-subdomain-out)
# ============================================================
print("\n" + "="*60, "\nB. VALIDATION PAR SOUS-DOMAINES (niveau 2)\n", "="*60)
subdomain_results = []
sous_domaines = estuaire['sous_domaine'].unique()
for target in TARGETS:
    sub = estuaire.dropna(subset=[target]).copy()
    y_full = (sub[target] == 1).astype(int)
    for sd in sous_domaines:
        train_mask = sub['sous_domaine'] != sd
        test_mask = sub['sous_domaine'] == sd
        if test_mask.sum() < 50:
            continue
        pipe = Pipeline([('prep', make_preprocess()),
                          ('clf', XGBClassifier(random_state=SEED, eval_metric='logloss', n_jobs=-1, **BEST_XGB))])
        pipe.fit(sub.loc[train_mask, FEATURES], y_full[train_mask.values])
        proba = pipe.predict_proba(sub.loc[test_mask, FEATURES])[:, 1]
        y_te = y_full[test_mask.values]
        taux_reel = y_te.mean()
        taux_predit = proba.mean()
        subdomain_results.append({
            'cible': target, 'sous_domaine': sd, 'n': int(test_mask.sum()),
            'taux_reel': taux_reel, 'taux_predit': taux_predit,
            'erreur_pts': (taux_predit - taux_reel) * 100,
        })
        print(f"  {target} / {sd[:40]:40s} n={int(test_mask.sum()):6d} erreur={(taux_predit-taux_reel)*100:+.1f}pts")
    pd.DataFrame(subdomain_results).to_csv(DATA_DIR / "resultats_niveau2_sousdomaines.csv", index=False)

sd_df = pd.DataFrame(subdomain_results)
print("\nDispersion de l'erreur par cible (marge d'incertitude niveau structure) :")
for target in TARGETS:
    e = sd_df[sd_df['cible'] == target]['erreur_pts']
    print(f"  {target}: min={e.min():+.1f} max={e.max():+.1f} std={e.std():.2f} pts (marge = ±{e.abs().max():.1f}pts, pire cas)")

# ============================================================
# C. INCERTITUDE IRREDUCTIBLE -- taux conditionnel par profil categoriel
# ============================================================
print("\n" + "="*60, "\nC. INCERTITUDE IRREDUCTIBLE (plancher bayesien approche)\n", "="*60)
irred_results = []
for target in TARGETS:
    sub = estuaire.dropna(subset=[target]).copy()
    sub['y'] = (sub[target] == 1).astype(int)
    sub['profil'] = sub[CATEGORICAL].astype(str).agg('|'.join, axis=1)
    grp = sub.groupby('profil')['y'].agg(['mean', 'count'])
    grp = grp[grp['count'] >= 30]  # profils suffisamment peuples
    grp['bayes_err_profil'] = np.minimum(grp['mean'], 1 - grp['mean'])
    bayes_err_pondere = (grp['bayes_err_profil'] * grp['count']).sum() / grp['count'].sum()
    n_profils = len(grp)
    couverture = grp['count'].sum() / len(sub)
    irred_results.append({'cible': target, 'n_profils': n_profils, 'couverture': couverture,
                           'erreur_bayes_approchee': bayes_err_pondere})
    print(f"{target}: {n_profils} profils (>=30 obs, {couverture:.1%} de la base) -- "
          f"erreur bayésienne approchée = {bayes_err_pondere:.3f} "
          f"(plancher d'erreur si le modèle ne connaissait que ces variables catégorielles)")

pd.DataFrame(irred_results).to_csv(DATA_DIR / "resultats_incertitude_irreductible.csv", index=False)
print("\nTerminé.")
