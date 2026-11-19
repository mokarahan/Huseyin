# AI-Doktora | Battery SOH & RUL Research Laboratory

**Reconstruction scientifique du projet NASA Battery — Windows / Python 3.9**  
**Statut :** pipeline de recherche exécutable et testé sur des données synthétiques. Les scores NASA ne sont **pas** fournis, faute de fichiers `.mat` NASA dans l'archive initiale.  
**Langue du guide :** français. Code et variables documentés en anglais technique court.

---

## 1. Question scientifique : que prédit exactement ce projet ?

Une cellule lithium-ion voit sa capacité décroître au fil de ses décharges. Le but n'est ni de classifier « bonne/mauvaise batterie », ni de retrouver directement une capacité déjà fournie en entrée, mais d'exploiter **uniquement les observations disponibles après les décharges passées** pour produire deux estimations :

1. **SOH du prochain cycle** : santé de la cellule lors de la *prochaine* décharge, exprimée en % de la capacité nominale.
2. **RUL au cycle observé** : nombre de décharges encore nécessaires avant le **premier** franchissement de 80 % de la capacité nominale.

Cette version donne une réponse rigoureuse à un point de la documentation historique : l'ancien README évoquait seulement le RUL avec un seuil de 70 %, tandis que les notes méthodologiques `NASA_Batarya_SOH_RUL_Tahmin_Yontemi` décrivent clairement la prédiction jointe SOH futur + RUL et un seuil commun de 80 %. Le protocole de ce projet adopte **80 %**, fixe et déclaré avant les expériences, parce que le franchissement serait observable pour B0005, B0006 et B0007 selon ces notes. Cela doit être vérifié automatiquement sur les vrais fichiers.

### Définition physique des cibles

Pour la décharge d'indice `k` (indices internes de 0 à N-1), `C_k` représente la capacité mesurée, et `C_nom = 2.0 Ah` :

```text
SOH(k+1) [%] = 100 × C_(k+1) / C_nom
k_EOL       = min { j : C_j <= 1.6 Ah }
RUL(k)      = k_EOL - k                         [décharges restantes]
```

Une observation d'entraînement se termine au cycle `k`, avec `k < k_EOL`, et n'utilise **aucune mesure du cycle `k+1` en entrée**. Elle porte une cible future SOH(k+1) et une cible RUL(k). Comme ces cibles ont des unités différentes, elles sont normalisées **séparément sur l'entraînement**.

**Précision sur le RUL = 0.** Pour préserver des cibles communes SOH(k+1) et RUL(k) jusqu'à la limite EOL, le pipeline s'arrête au cycle qui précède le franchissement. Le cycle EOL à RUL = 0 n'est donc pas une observation d'entraînement multitâche. Il reste défini mathématiquement; cela doit être indiqué lors de toute publication.

**Précision opérationnelle :** la courbe d'une décharge doit être **terminée** avant la prédiction. Ce n'est pas une prédiction « à tout instant » à bord d'un véhicule. L'extrapolation à des mesures partielles ou à un BMS réel n'est pas prouvée.

## 2. Dataset : NASA PCoE, pas un jeu inventé

- Source officielle : **B. Saha et K. Goebel (2007), Battery Data Set, NASA Ames Prognostics Data Repository**.
- URL archive : https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip
- Portail NASA : https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- Cellules ciblées : `B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat` si les données sont présentes.
- Trois signaux : `Voltage_measured`, `Current_measured`, `Temperature_measured`.
- `Time` permet le rééchantillonnage et la durée physique; `Capacity` sert **exclusivement** à fabriquer et à contrôler les étiquettes, pas à alimenter le réseau.
- Les opérations de charge et d'impédance sont exclues dans ce protocole; les utiliser demanderait une nouvelle hypothèse expérimentale.

**L'archive de départ ne contenait AUCUN fichier `.mat`.** Il serait scientifiquement incorrect de créer des fichiers ressemblant à des mesures NASA. Le présent ZIP inclut donc : un téléchargeur pointant vers NASA, un importeur MAT, un audit des données, et un générateur de **données synthétiques pour les tests logiciels seulement**. Les fichiers bruts officiels restent à récupérer sur la machine de l'utilisateur.

## 3. Installation Windows propre (Python 3.9)

### Prérequis

- Windows 10/11, 64 bits.
- Python **3.9.x 64 bits** avec le lanceur `py.exe` : https://www.python.org/downloads/release/python-3913/ (ou une autre 3.9.x).
- Connexion Internet pour installer les packages et récupérer NASA (sauf utilisation d'une archive officielle déjà téléchargée).
- RAM de 8 Go recommandée; entraînement **CPU** possible. CUDA optionnel, si PyTorch et les pilotes sont compatibles.

Dans PowerShell, ouvert **dans le dossier décompressé** :

```powershell
# Évite de changer les politiques d'exécution globales de Windows.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\win\installer.ps1
```

L'installation crée `.venv` en utilisant Python 3.9 puis lance les tests. Si `py -3.9` n'existe pas, installez Python 3.9 64 bits avant de poursuivre. Ne copiez pas un interpréteur Python à la main dans le projet : un environnement virtuel est plus sûr et maintenable.

Les dépendances sont épinglées dans `requirements.txt` avec des versions compatibles Python 3.9. Les GPU récents peuvent nécessiter d'adapter **uniquement** la ligne `torch` selon la matrice officielle de compatibilité PyTorch.

### Obtenir les vraies données

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\win\nasa.ps1
```

Le téléchargeur ouvre la source **officielle** NASA, conserve l'archive sous `data/raw/`, cherche les `.mat` dans ses ZIP internes, extrait les cellules retenues et écrit `data/raw/provenance.json` avec les sommes SHA-256. La préparation construit les fichiers `data/processed/Bxxxx.npz` et `data/processed/audit.json`.

**Mode sans téléchargement automatique :** télécharger vous-même l'archive officielle, puis :

```powershell
.\.venv\Scripts\python.exe main.py download --archive "D:\donnees\5. Battery Data Set.zip"
.\.venv\Scripts\python.exe main.py prepare
.\.venv\Scripts\python.exe main.py audit
```

Si la source NASA ne répond pas, **ne pas substituer automatiquement un miroir inconnu**. Se procurer l'archive par une source fiable, contrôler sa provenance et utiliser `--archive`. Les trois dossiers `data/raw`, `data/processed`, `results` n'ont pas besoin de chemins absolus, `HOME`, WAMP ou du terminal Linux.

### Exécuter la vérification et une première expérience

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests
.\.venv\Scripts\python.exe main.py audit
.\.venv\Scripts\python.exe main.py train --model hybrid --test B0005 --epochs 100 --window 10 --seed 42
```

**Comparer les méthodes**, chacune avec les mêmes groupes de batteries et la même fenêtre :

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\win\comparer.ps1 -Epochs 100 -Window 10 -Seed 42
```

Ou, directement :

```powershell
.\.venv\Scripts\python.exe main.py compare --test all --epochs 100 --window 10 --seed 42
```

`--test all` prend tour à tour chaque batterie admissible comme **batterie de test entière**, en garde une autre en validation et les batteries restantes en apprentissage. Avec trois cellules, il n'y a qu'**une cellule d'entraînement** par fold : les performances seront donc incertaines. Avec quatre cellules admissibles, il y en a deux. Chaque cellule test conserve son indépendance.

### Lire une prédiction

```powershell
.\.venv\Scripts\python.exe main.py predict `
  --checkpoint "results\experiments\seed_42\B0005\hybrid\model.pt" `
  --battery B0005 --cycle 30
```

La commande produit un nombre réel de cycles restants et le SOH (%) de la prochaine décharge. La prédiction RUL **brute** est conservée même si elle est négative; une seconde valeur, bornée à 0, est prévue **uniquement pour l'affichage**. Ne pas recalculer le score en utilisant cette valeur bornée comme si elle était la prédiction brute.

### Démonstration purement technique (sans NASA)

Si vous souhaitez vérifier le code avant de télécharger les données, dans un **dossier propre** où `data/processed/` ne contient aucun `B*.npz` :

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\win\demo.ps1
```

**ATTENTION :** ceci crée des signaux de laboratoire **synthétiques**, avec l'avertissement `DEMO_NOT_REAL_DATA.txt`. AUCUN score, graphique ou poids issu de la démonstration ne doit être présenté comme performance sur NASA. Avant l'étude réelle, déplacer/effacer les `.npz` de démonstration et leur avertissement, puis exécuter `download` et `prepare`.

## 4. Architecture étudiée et fonctions d'activation

La variante hybride est **CNN → MHA causale → TCN causale → FCL → 2 têtes linéaires**. L'ordre suit les notes méthodologiques associées au ZIP reçu.

```text
W cycles de décharge achevés     X : [batch, W, 128, 3]       A : [batch, W, 2]
             │                              │                          │
             │                         CNN 1D par cycle               │
             │                     Conv(3→16,k=5), ReLU                │
             │                     MaxPool(2), Conv(16→32,k=3), ReLU   │
             │                     AdaptiveAvgPool(4), Linear(128→32) │
             │                              └──────────┬───────────────┘
             │                          Fusion+ReLU [batch,W,32]
             │                                      │
             │                    MHA causale 4 têtes, résidu, LayerNorm
             │                                      │
             │                   TCN résiduel causal 2 blocs (d=1,2)
             │                                      │
             └──────────────────── Dernier cycle observable ──────────┘
                                                    │
                                      Linear(32→16), ReLU, Dropout
                                            ┌───────┴────────┐
                                      Linear(16→1)     Linear(16→1)
                                      SOH(k+1)            RUL(k)
                                        Z-score           Z-score
```

L'entrée auxiliaire a **deux variables observables** : `log(1 + durée de décharge en secondes)` et **rang de décharge achevée**. La durée absolue serait perdue après le rééchantillonnage temporel sur [0,1]. Le rang de décharge est connu au moment de l'estimation, mais peut produire un raccourci statistique; son retrait doit faire l'objet d'une expérience de sensibilité avant toute conclusion sur l'architecture.

| Couche / activation | Pourquoi ici ? |
|---|---|
| CNN1D à 2 convolutions, noyaux 5 et 3 | Détecter les variations locales des trois signaux sans imposer un grand réseau sur de petites séries. |
| ReLU après convolutions et projection | Non-linéarité peu coûteuse, simple à reproduire; hypothèse à tester face à GELU si nécessaire. |
| Pooling 2 puis pooling adaptatif 4 | Réduire les mesures 128 points en une représentation compacte, sans fixer les valeurs du batch. |
| MHA 4 têtes, dimension 32 | Apprendre des relations entre les cycles passés; le masque triangulaire supérieur interdit d'examiner les cycles ultérieurs. |
| TCN 2 blocs, dilatations 1 et 2 | Capter des tendances sur plusieurs cycles avec convolutions causales; réceptif théorique de 7 cycles après les deux blocs (noyau 3), sans promettre qu'il résume toute la fenêtre W. |
| FCL `32→16`, ReLU, dropout 0.10 | Petite tête partagée pour limiter la sur-paramétrisation. |
| 2 sorties **linéaires**, sans activation | Les **cibles z-score peuvent être négatives**; ReLU/Softplus seraient inadaptées pendant l'entraînement. |

**Pourquoi pas 8 têtes, 6 couches ou un grand Transformer ?** Le protocole de départ concerne trois à quatre cellules, pas des milliers de vieillissements indépendants. Les fenêtres chevauchantes ne créent pas de nouvelles cellules indépendantes. Une architecture plus lourde nécessite une justification expérimentale et un budget de validation crédible, pas seulement une capacité théorique supérieure.

## 5. Préparation sans fuite d'information

1. Charger les opérations `discharge` valides depuis les fichiers `.mat`.
2. Vérifier temps, courant, tension, température et capacité; écrire toutes les exclusions.
3. Trier les instants et rééchantillonner **chaque décharge complète** à 128 points sur son propre temps relatif.
4. Déterminer **l'EOL observé par cellule** au seuil fixe de 1,6 Ah; si jamais atteint, indiquer `censored` et **ne pas inventer** de RUL exact.
5. Construire les séquences de W décharges qui se terminent au cycle k, jamais au-delà.
6. Séparer **les cellules**, avec un rôle exclusif entraînement/validation/test.
7. Ajuster **les trois standardisations** : signaux (3 canaux), auxiliaires (2 variables), cibles (SOH et RUL), sur les **seules fenêtres train**.
8. Appliquer ces mêmes paramètres, sans recalcul, aux fenêtres de validation et de test.
9. Enregistrer les unités réelles, le rang de chaque prédiction et les résultats par batterie.

L'étiquette SOH(k+1) utilise une donnée future uniquement dans la **vérité terrain**, jamais comme feature. La capacité courante ou future n'est pas concaténée au vecteur d'entrée. La seule manière de vérifier la généralisation aux cellules inconnues reste la séparation stricte des cellules au test.

## 6. Évaluation : que comparer et que publier ?

Les modèles disponibles sont :

| Nom CLI | Modèle | Question contrôlée |
|---|---|---|
| `ridge` | Régression ridge, descriptifs des W cycles | La simplicité suffit-elle ? |
| `cnn` | CNN du cycle le plus récent | Faut-il même un historique séquentiel ? |
| `cnn_lstm` | CNN + LSTM une couche | Récurrence adaptée aux petites données ? |
| `cnn_tcn` | CNN + TCN deux blocs | Valeur de la convolution temporelle sans attention ? |
| `hybrid` | CNN + MHA causale + TCN | L'attention apporte-t-elle vraiment mieux que CNN-TCN ? |

Même fenêtre, même graine et mêmes groupes pour chaque comparaison. L'apprentissage des réseaux utilise **AdamW** (taux `1e-3`, pénalité `1e-4`), des gradients bornés et l'arrêt anticipé selon la seule validation. Les sorties sont normalisées séparément. La fonction de perte est :

```text
L = 0.5 × MSE( zSOH_pred, zSOH_true )
  + 0.5 × SmoothL1( zRUL_pred, zRUL_true ; beta=1 )
```

MSE pénalise les grandes erreurs SOH; SmoothL1/Huber limite l'influence des très grandes erreurs RUL. La pondération 0,5 / 0,5 est **un choix de départ**, pas une valeur optimisée. Il faut tester la sensibilité sur les batteries de développement, jamais sur les résultats test.

**Scores primaires :** MAE et RMSE du RUL en **cycles** et du SOH en **points de pourcentage**. R² SOH est fourni comme complément. Les figures affichent vérité terrain et prédiction **brute** dans les mêmes unités. Avec seulement 3 ou 4 cellules, préférer les résultats **par cellule** et la moyenne **macro par cellule**, sans pseudo-significativité calculée sur les fenêtres corrélées.

**Ne pas annoncer une amélioration SOTA** à partir de ce projet seul. Il faudra évaluer plusieurs graines (`42, 123, 2027` par exemple), quantifier la variabilité, comparer des références implémentées sur les mêmes données, puis, idéalement, confirmer sur une cohorte externe (CALCE/MATR) sans changer le protocole en fonction du test.

## 7. Dossier et responsabilités des fichiers

```text
AI_Doktora_RUL_Windows/
├── README.md                 But, lancement, hypothèses et interprétation
├── requirements.txt          Versions Python 3.9 fixées
├── main.py                   Entrée CLI (prepare/train/compare/predict)
├── src/
│   ├── config.py             Constantes EOL, modèle et entraînement
│   ├── data.py               Lecture MATLAB, filtrage, cibles, fenêtres
│   ├── download.py           Acquisition officielle et SHA-256
│   ├── scaling.py            Standardisation TRAIN uniquement
│   ├── models.py             CNN, LSTM, TCN, attention, hybride
│   ├── experiment.py         Validation par cellule, pertes et métriques
│   ├── plots.py              Graphiques réels en PNG depuis le CSV
│   └── demo.py               Données synthétiques de TEST uniquement
├── data/
│   ├── raw/                 .mat NASA + provenance (après download)
│   └── processed/           .npz + audit (après prepare)
├── win/
│   ├── installer.ps1         Environnement Python 3.9
│   ├── nasa.ps1              Acquisition + conversion
│   ├── comparer.ps1          Baselines et ablations
│   └── demo.ps1              Test logiciel synthétique
├── tests/                    Tests unitaires et intégration
├── docs/
│   └── PROTOCOLE_SCIENTIFIQUE.md   Limites, décisions, audit et références
└── results/                  Sorties expérimentales, non versionnées
```

### Sorties d'un fold

```text
results/experiments/
  comparison.csv                métriques par cellule et variante
  comparison_summary.json       moyenne et écart-type macro inter-cellules
  seed_42/B0005/hybrid/
    model.pt                    poids + configuration + scaler TRAIN
    report.json                 groupes, paramètres, temps, métriques
    history.csv                 pertes train / validation par époque
    predictions.csv             chaque cycle: vrai, prédit et unité
    soh_prediction.png          SOH futur vrai / prédit
    rul_prediction.png          RUL vrai / prédit
```

Le modèle Ridge ne produit pas `model.pt` : ses coefficients sont enregistrés en `.npz`, sans mécanisme de chargement Ridge pour `predict`. La commande `predict` s'applique aux quatre modèles PyTorch seulement.

## 8. Auto-évaluation et limites actuellement connues

**Faits vérifiés sur la livraison :** les tests automatiques passent et les modèles s'entraînent et prédisent avec des entrées **synthétiques**. Les `.mat` officiels ne sont pas fournis; leur import ne pourra être considéré comme validé bout en bout que lorsque les archives NASA auront été téléchargées et contrôlées. Le téléchargement peut dépendre de la disponibilité du serveur NASA et de la connexion locale.

**Limites scientifiques non résolues :** cohorte minuscule, forte dépendance entre fenêtres, protocoles de décharge et tensions d'arrêt différents, dépendance au rang de cycle, définition SOH(k+1) et RUL(k) commune mais imparfaite au cycle EOL, absence d'intervalles d'incertitude et de validation sur BMS. Le seuil 80 % est un **critère expérimental** de ce protocole, pas une règle universelle de sécurité batterie.

**GWO (Grey Wolf Optimizer) :** évoqué dans les notes historiques, **non implémenté ici**. Son coût, son espace de recherche et son intérêt doivent être comparés à une recherche aléatoire sur les seules cellules de développement. Ajouter un optimiseur sophistiqué avant d'avoir démontré l'utilité de l'attention est une source de complexité injustifiée.

Pour tout rapport académique, lire le document `docs/PROTOCOLE_SCIENTIFIQUE.md` avant d'interpréter les résultats.

## 9. Références sélectionnées

- Saha, B. & Goebel, K. (2007), *Battery Data Set*, NASA Ames PCoE Repository : https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- Vaswani et al. (2017), *Attention Is All You Need*, NeurIPS : https://arxiv.org/abs/1706.03762
- Bai, Kolter & Koltun (2018), *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling* : https://arxiv.org/abs/1803.01271
- Zhang (2025), *Robustness of CNN-augmented sequential models for Li-ion battery RUL prediction under data scarcity*, PLOS ONE : https://doi.org/10.1371/journal.pone.0339528
- *Lithium-ion battery RUL prediction based on optimized VMD-SSA-PatchTST algorithm* (2025), Scientific Reports : https://www.nature.com/articles/s41598-025-11934-7
- Zhu et al. (2026), *Leakage-Resistant and Censor-Aware Evaluation of Capacity-Only Online Remaining-Useful-Life Prediction...*, Batteries : https://doi.org/10.3390/batteries12110484

**Périmètre des références :** elles justifient des précautions et des mécanismes possibles. Elles ne prouvent pas que la variante hybride de ce dépôt surpasse la littérature. Des protocoles, métriques et seuils différents empêchent les comparaisons numériques directes.
