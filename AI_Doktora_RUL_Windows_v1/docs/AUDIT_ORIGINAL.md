# Audit de l'archive AI-Doktora-main.zip reçue

**Objet :** justification vérifiable des modifications du projet, sans confondre code original, intention scientifique et fonctionnalités confirmées.

## Sources examinées

- `README.md` et `requirements.txt` historiques;
- `src/DataExporter.py`, `src/DataNormalizer.py`, `src/dataset/BatteryDataset.py`;
- `src/models/{CNN,LSTM,TCN,MHA}.py` et les quatre `*Handler.py`;
- scripts Windows PowerShell et Linux Shell;
- notes fournies : `docs/pdf/001-NASA_Batarya_SOH_RUL_Tahmin_Yontemi.pdf`, `002-NASA_128_Nokta_Egitim_Notu.pdf`, `003-NASA_Batarya_Z_Score_Standardizasyon.pdf`.

## Constats précis

| Constat dans l'archive initiale | Conséquence | Décision dans la reconstruction |
|---|---|---|
| README : RUL sur NASA, seuil 70 % (1,4 Ah); note 001 : deux cibles SOH futur / RUL, seuil 80 % (1,6 Ah). | Les questions scientifiques et les labels ne décrivent pas exactement la même étude. | Objectif commun défini dans README, seuil 80 % fixé et audité. |
| Aucun `.mat` dans les 92 entrées du ZIP. | Entraînement NASA non exécutable à partir de cette archive seule. | Téléchargeur officiel + import contrôlé, aucune donnée NASA inventée. |
| `DataNormalizer.py` utilise `EOL_IDX` sans garantie qu'un cycle franchisse le seuil. | Risque d'exception et de faux label sur une cellule censurée. | `eol_index` retourne `None` et exclut les cellules censurées du supervisé. |
| `DataNormalizer.py` calcule des extrema globaux durant l'itération des cellules. | Risque de contaminer un prétraitement par les cellules test; confusion MinMax et z-score. | Trois transformations z-score calculées après split, sur train uniquement. |
| `BatteryDataset.py` applique un z-score individuel aux colonnes du CSV. | Les décalages physiques entre cycles peuvent être effacés, contrairement à la note 003. | Standardisation globale par canal entraînement; durées préservées. |
| Le RUL était encodé dans le **nom du fichier CSV**. | Métadonnées fragiles, mélange identifiants et labels; risque d'erreur dans le tri et l'évaluation. | `.npz` et matrices étiquettes explicites, numéro de cycle sauvegardé. |
| Le modèle CNN consomme `[B,128,4]` mais le préprocesseur de la note 002 renvoie `[128,3]` V/I/T. | Incohérence entre canaux réellement informatifs et nombre d'entrées. | Signaux V/I/T = 3; durée/rang = 2 auxiliaires séparés. |
| La classe TCN historique spécifie `input_shape='NLC'` et sélectionne `tcn_out[:,:, -1]`. | Risque de sélectionner une **dimension de canal** plutôt que le dernier pas temporel, selon le contrat du package. | TCN native `Conv1d`, axes documentés, dernier pas lu explicitement. |
| Attention historique paramétrée avec `pos_embedding(1,128,d_model)`. | Couplage rigide à 128 pas internes et confusion entre échantillons d'un cycle et cycles historiques. | Attention après encodage par cycle, séquence de W cycles, masque causal. |
| Quatre classes `*Handler` répètent presque toute la boucle de train/validation/test. | Maintenance pénible et divergences potentielles de métriques/early stopping. | Une boucle de train / un pipeline de métriques / une usine de variantes. |
| Scripts référencent `HOME/Dev/AI-Doktora` et des chemins de venv personnels. | Exécution Windows non portable. | Scripts `win/*.ps1` calculant le dossier racine via `$PSScriptRoot`. |
| La note 001 envisage GWO sans expériences le justifiant. | Recherche d'hyperparamètres coûteuse, risque de sélection sur test. | GWO réservé à une étude avec budget et controls explicites, non ajouté artificiellement. |

## Périmètre de vérification

La reconstruction possède des tests de forme, de génération d'étiquettes, de rééchantillonnage, de causalité MHA/TCN, de normalisation, de séparation des cellules, de lecture MATLAB synthétique, d'extraction des ZIP imbriqués, ainsi qu'un essai complet avec données synthétiques.

**Ce que l'audit ne prétend pas :** validation empirique du modèle sur de vraies courbes NASA, reproduction des résultats des PDF et preuve de supériorité de la MHA. Ces conclusions exigent l'exécution sur les fichiers officiels, un audit des exclues, plusieurs folds et un protocole statistique approprié.
