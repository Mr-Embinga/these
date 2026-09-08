# -*- coding: utf-8 -*-
"""
Enrichit les artefacts de l'application : agrégats provinciaux avec taux CORRIGÉ (pas seulement
naïf), arbitrage documenté par réseau, importance des variables du modèle retenu, et synthèse des
résultats de validation pour la page Méthode.

Arbitrage (décision actée le 2026-08-15, note_correction_estimations.md §4) :
  - Eau, électricité : taux repondéré retenu (bénéfice net mesuré, -18% et -36% d'erreur moyenne).
  - Fibre : taux naïf conservé (repondération inefficace en moyenne, -4%, et dégradée sur 3
    provinces) -- affiché avec avertissement sur le plancher d'incertitude irréductible (36,2%).
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(r"C:\Users\hp\Desktop\MVP\these\pipeline\data")
MODEL_DIR = Path(r"C:\Users\hp\Desktop\MVP\these\pipeline\models")

PROVINCES = {2: 'Haut-Ogooué', 3: 'Moyen-Ogooué', 4: 'Ngounié', 5: 'Nyanga',
             6: 'Ogooué-Ivindo', 7: 'Ogooué-Lolo', 8: 'Ogooué-Maritime', 9: 'Woleu-Ntem'}
TARGET_LABELS = {'fdsm123': 'eau', 'fdsm124': 'électricité', 'fdsm125': 'fibre'}
CIBLES_CORRIGEES = {'fdsm123', 'fdsm124'}  # fdsm125 (fibre) reste au taux naif, cf. arbitrage ci-dessus

niveau3 = pd.read_csv(DATA_DIR / "resultats_niveau3_par_province.csv")
pond = pd.read_csv(DATA_DIR / "resultats_repondération.csv")
sousdomaines = pd.read_csv(DATA_DIR / "resultats_niveau2_sousdomaines.csv")
decalage = pd.read_csv(r"C:\Users\hp\Desktop\MVP\dist\data\decalage_distributions_par_province.csv")
irreductible = pd.read_csv(DATA_DIR / "resultats_incertitude_irreductible.csv")
calibration = pd.read_csv(DATA_DIR / "resultats_calibration.csv")

marge_par_cible = sousdomaines.groupby('cible')['erreur_pts'].apply(lambda s: s.abs().max()).to_dict()
SEUIL_RESTITUTION = 10

agregats = []
for target, reseau in TARGET_LABELS.items():
    for pcode, pname in PROVINCES.items():
        n3 = niveau3[(niveau3['cible'] == target) & (niveau3['province'] == pname)]
        pd_row = pond[(pond['cible'] == target) & (pond['province'] == pname)]
        dec_row = decalage[decalage['province'] == pname]
        n = int(n3['n'].iloc[0]) if len(n3) else None

        taux_naif = float(n3['taux_predit_naif'].iloc[0]) if len(n3) else None
        taux_corrige = float(pd_row['taux_pondere'].iloc[0]) if len(pd_row) else None
        correction_appliquee = target in CIBLES_CORRIGEES and taux_corrige is not None
        taux_retenu = taux_corrige if correction_appliquee else taux_naif

        agregats.append({
            'cible': target, 'reseau': reseau, 'province': pname, 'n': n,
            'taux_naif': taux_naif, 'taux_corrige': taux_corrige,
            'correction_appliquee': correction_appliquee, 'taux_retenu': taux_retenu,
            'marge_incertitude_pts': float(marge_par_cible.get(target, float('nan'))),
            'ecart_composition_jsd': float(dec_row['JSD_moyenne'].iloc[0]) if len(dec_row) else None,
            'restituable': (n or 0) >= SEUIL_RESTITUTION,
        })

df_agregats = pd.DataFrame(agregats)
df_agregats.to_csv(MODEL_DIR / "agregats_provinciaux.csv", index=False)
print(f"Agrégats enrichis sauvegardés : {MODEL_DIR / 'agregats_provinciaux.csv'}")

# ---- importance des variables (modele retenu, XGBoost) ----
CATEGORICAL = ['fdsm121', 'fdsm111', 'fdsm202', 'fdsm122', 'fdsm201', 'cl108a', 'cl108b']
CONTINUOUS = ['fdsm126', 'fdsm127', 'fdsm128', 'fdsm213', 'fdsm214', 'fdsm215', 'fdsm216', 'fdsm217', 'fdsm218']
LABELS = json.load(open(MODEL_DIR / "labels.json", encoding="utf-8"))
LABEL_CONTINU = {
    'fdsm126': 'Total logements', 'fdsm127': 'Total ménages', 'fdsm128': 'Population',
    'fdsm213': 'Personnes du ménage', 'fdsm214': 'Femmes du ménage', 'fdsm215': 'Hommes du ménage',
    'fdsm216': 'Personnes <5 ans', 'fdsm217': 'Personnes 5-15 ans', 'fdsm218': 'Personnes 16+ ans',
}

importances_toutes = []
for target, reseau in TARGET_LABELS.items():
    pipe = joblib.load(MODEL_DIR / f"modele_{target}.joblib")
    prep = pipe.named_steps['prep']
    clf = pipe.named_steps['clf']
    noms_colonnes = list(prep.named_transformers_['cat'].get_feature_names_out(CATEGORICAL)) + CONTINUOUS
    imp = pd.Series(clf.feature_importances_, index=noms_colonnes)

    # regroupement par variable source (somme des categories one-hot) pour lisibilite
    imp_par_variable = {}
    for c in CATEGORICAL:
        imp_par_variable[LABELS[c]['label']] = imp[[i for i in imp.index if i.startswith(c + '_')]].sum()
    for c in CONTINUOUS:
        imp_par_variable[LABEL_CONTINU[c]] = imp.get(c, 0.0)

    top5 = pd.Series(imp_par_variable).sort_values(ascending=False).head(5)
    for var, val in top5.items():
        importances_toutes.append({'reseau': reseau, 'variable': var, 'importance': round(float(val), 4)})

pd.DataFrame(importances_toutes).to_csv(MODEL_DIR / "importance_variables.csv", index=False)
print(f"Importance des variables sauvegardée : {MODEL_DIR / 'importance_variables.csv'}")

# ---- synthese pour la page Methode ----
synthese = {
    'auc_niveau1': {'eau': 0.760, 'électricité': 0.904, 'fibre': 0.698},
    'auc_niveau3_hors_domaine': {'eau': 0.790, 'électricité': 0.862, 'fibre': 0.694},
    'erreur_absolue_moyenne_naive_pts': {'eau': 4.39, 'électricité': 10.22, 'fibre': 10.44},
    'erreur_absolue_moyenne_corrigee_pts': {'eau': 3.60, 'électricité': 6.55, 'fibre': 10.04},
    'ece_interne': calibration.set_index('cible')['ece_interne_estuaire'].round(4).to_dict(),
    'ece_hors_domaine': calibration.set_index('cible')['ece_externe_autres_provinces'].round(4).to_dict(),
    'erreur_bayes_irreductible': irreductible.set_index('cible')['erreur_bayes_approchee'].round(3).to_dict(),
    'design_multiprovinces_erreur_moyenne_pts': {'eau': 3.72, 'électricité': 2.57, 'fibre': 5.89},
    'design_estuaire_pur_erreur_moyenne_pts': {'eau': 3.49, 'électricité': 6.82, 'fibre': 10.40},
}
with open(MODEL_DIR / "synthese_methode.json", "w", encoding="utf-8") as f:
    json.dump(synthese, f, ensure_ascii=False, indent=2)
print(f"Synthèse méthode sauvegardée : {MODEL_DIR / 'synthese_methode.json'}")
