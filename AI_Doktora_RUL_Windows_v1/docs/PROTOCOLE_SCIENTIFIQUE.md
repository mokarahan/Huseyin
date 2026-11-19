# Protocole scientifique vérifiable — AI-Doktora

## Positionnement et objet

Ce projet ne revendique pas l'invention du CNN, du mécanisme de self-attention, du TCN ou de leur enchaînement. Il examine une **hypothèse falsifiable** : sur les données de décharge NASA disponibles, une représentation locale CNN suivie d'attention causale entre cycles et d'une convolution temporelle causale permet-elle une meilleure prédiction **hors batterie** du SOH futur et du RUL qu'un modèle plus simple ?

La conclusion « la combinaison CNN–MHA–TCN est utile » n'est justifiée **que si** sa performance hors batterie et sa stabilité dépassent celles des contrôles pertinents. Une performance moins bonne constitue un résultat scientifique valide; la combinaison ne doit pas être défendue a priori.

## Définition formelle

Soit une cellule `b` avec `N_b` cycles de décharge valides observés, triés dans leur ordre expérimental. Pour le cycle `k` :

- `V_{b,k}(t), I_{b,k}(t), T_{b,k}(t)` : mesures sur une décharge achevée;
- `C_{b,k}` : capacité mesurée durant cette décharge (Ah);
- `D_{b,k}` : durée en secondes, extraite de la série temporelle;
- `C_nom = 2 Ah` : capacité nominale dans ce protocole;
- `C_EOL = 0.8 × C_nom = 1.6 Ah` : seuil EOL fixé avant le test.

Le premier indice de franchissement observé est :

`e_b = min{k | C_{b,k} <= C_EOL}`.

Il peut ne pas exister : une cellule sans franchissement est **censurée à droite**. Le dernier cycle enregistré d'une cellule censurée **n'est pas un événement EOL** et ne reçoit jamais artificiellement un label RUL=0.

Pour chaque `k < e_b`, le vecteur cible est :

`y_(b,k) = (100 × C_(b,k+1)/C_nom, e_b - k)`.

**Important :** une décharge est une unité d'observation; une valeur RUL de 20 signifie « 20 décharges restantes jusqu'au premier franchissement expérimental du seuil », pas 20 heures ni 20 jours. La future SOH à `k+1` ne doit jamais apparaître dans l'entrée.

L'entrée de chaque exemple est :

`X_(b,k) = [ S_(b,k-W+1), …, S_(b,k) ] ∈ R^(W×128×3)`.

`S` représente les signaux V/I/T interpolés sur 128 fractions du temps de décharge **propre à ce cycle**, sans utiliser les cycles futurs. Le complément `A_(b,k) ∈ R^(W×2)` comprend les durées observées (`log1p`) et les rangs de décharge. **Aucune capacité ne figure dans X ou A.**

### Caractérisation du délai et de la causalité

Chaque exemple est construit juste après la décharge `k`. `D_(b,k)` est alors observable, car la courbe entière est terminée. Le modèle ne peut pas être utilisé en temps réel pendant une décharge partielle sans adaptation et validation nouvelles.

Pour les couches de séquence, toute position future `j > i` est masquée dans l'attention et ignorée par la convolution temporelle causale. Le prédicteur final exploite seulement la représentation du cycle `k`. Le masquage seul ne corrige **pas** une fuite causée ailleurs (label, normalisation, sélection de modèles).

## Chaîne et choix exacts de couches

### CNN par cycle

L'encodeur partage ses poids pour tous les cycles :

1. Conv1D `in_channels=3`, `out_channels=16`, `kernel=5`, padding symétrique 2 dans le temps interne **du cycle**; ReLU.
2. MaxPool1D, taille 2 : 128 → 64 échantillons.
3. Conv1D 16→32, noyau 3, padding 1; ReLU.
4. AdaptiveAvgPool1D à 4 positions : 32 × 4 = 128 caractéristiques.
5. Linear 128→32, ReLU; concaténation avec les deux auxiliaires observables; Linear 34→32, ReLU.

Le CNN utilise l'ensemble du signal du **cycle achevé**. Il n'est pas causal **à l'intérieur** de cette décharge, ce qui est volontaire : la décision ne se produit qu'en fin de cycle. Il est causal **entre les cycles** puisque les cycles futurs sont absents de l'entrée.

**Pourquoi ReLU plutôt que sigmoid/tanh/GELU ?** ReLU réduit le risque de saturation et le coût par couche, et les deux convolutions suffisent comme point de départ. LSTM emploie ses portes internes sigmoid/tanh, imposées par la définition standard de la cellule. Les sorties SOH/RUL restent linéaires car leurs valeurs standardisées ne sont pas contraintes à être positives.

### MHA entre les cycles (hybrid uniquement)

Dimension totale `d_model=32`, `num_heads=4`, `8` dimensions par tête; un seul bloc d'attention PyTorch. Le masque `M[i,j]=True` pour `j>i` empêche toute attention au cycle futur. La sortie est ajoutée par un lien résiduel puis normalisée par `LayerNorm`. Aucune position artificielle n'est ajoutée : l'ordre est fourni par le rang connu dans A et la séquence de cycles. Cette décision doit être ablatée dans une étude approfondie.

Pour chaque tête : `Attention(Q,K,V) = softmax(QK^T / sqrt(d_head) + mask) V`, avec poids appris sur les représentations passées. Ce mécanisme **peut apprendre** des liens distants; l'utiliser ne démontre pas automatiquement une interprétabilité biologique ou une amélioration du SOH.

### TCN temporel causal (cnn_tcn et hybrid)

Deux blocs résiduels. Dans chacun, Conv1D 32→32, noyau `3`, dilatation `d=1` puis `d=2`; padding à gauche uniquement, ReLU, dropout 0.10, addition résiduelle. Le champ réceptif temporel est de `1 + (3-1)(1+2) = 7` cycles. L'attention de la variante hybride peut, elle, intégrer l'ensemble des W cycles, si les paramètres appris l'exploitent.

### FCL et deux têtes

Représentation du dernier cycle → `Linear 32→16` → ReLU → Dropout 0.10. Deux projections distinctes `Linear 16→1` produisent SOH prochain cycle et RUL courant, chacun sous forme z-score. La déstandardisation se fait en sortie, avec moyennes et écarts-types de **train**, jamais ceux de test.

### Nombre de couches : justification et limites

Deux CNN, un MHA, deux TCN et une couche cachée FCL forment un point de départ **compact et vérifiable**, non un optimum démontré. Comparaisons pertinentes futures : 1 versus 2 convolutions CNN, 1 versus 2 blocs TCN, 0 versus 1 bloc MHA, `W∈{10,20,30}`. Chaque comparaison requiert une justification avant exploitation des résultats test. L'ajout de couches sans preuve est déconseillé.

## Perte, standardisation et optimisation

Pour chacune des cibles `j` et chaque signal `c`, la transformation est `z=(x - μ_train)/σ_train`. Les statistiques sont ajustées sur les observations TRAIN exclusivement; des fenêtres chevauchantes influencent naturellement leur distribution et doivent être déclarées. Aucune normalisation individuelle par cycle n'est appliquée, car elle pourrait effacer les variations de niveau liées à l'usure.

`L = α×MSE(zSOH_pred,zSOH_true) + (1-α)×Huber_δ=1(zRUL_pred,zRUL_true)`, avec `α=0.5` pour le protocole initial. AdamW (`lr=0.001`, `weight_decay=1e-4`), gradient clipping norme 1.0, mini-lots 32, graine 42, patience 15 et 100 époques **maximum**.

Le modèle restauré après early stopping est celui du meilleur score **validation**. Les valeurs test n'interviennent pas dans les paramètres, le scaling, le nombre d'époques, la sélection des architectures ni la sélection des hyperparamètres.

## Plan d'expérience obligatoire

### Phase 0 — Audit des vrais fichiers

Pour chaque cellule : nombre de décharges, capacité minimale/maximale, EOL observé ou non, nombre d'enregistrements exclus, dimensions, durée de chaque décharge, hashes SHA-256. Vérifier la cohérence physique des durées et la qualité des séries. Un chargement qui échoue ne doit pas déclencher la génération automatique de données synthétiques.

### Phase 1 — Contrôles simples

Comparer Ridge, CNN, CNN-LSTM, CNN-TCN. Les architectures lourdes ne sont admissibles que si elles apportent une amélioration non triviale face à ces références. Il faut aussi contrôler l'effet du **rang de cycle** et de la **durée** : les supprimer séparément dans une étude complémentaire afin de savoir ce qui porte réellement la performance.

### Phase 2 — Hypothèse du modèle hybride

Comparer `hybrid` versus `cnn_tcn` : l'unique ajout architectural majeur est la MHA (avec normalisation résiduelle). Documenter le coût : nombre de paramètres, temps CPU/GPU, temps de prédiction (mesure spécifique encore non intégrée), mémoire.

### Phase 3 — Validité hors batterie

Une cellule entière est test, une autre est validation, les autres forment train. Répéter les folds indépendants sur B0005, B0006, B0007 et B0018 *si EOL observé*. Évaluer en priorité MAE/RMSE RUL en cycles et MAE/RMSE SOH futur en points de pourcentage. Reporter chaque fold, puis les agrégats macro. Les fenêtres d'une même cellule ne sont pas des réplications indépendantes.

**Limite du protocole actuel :** pour chaque test, la cellule de validation est choisie par ordre alphabétique parmi les cellules restantes. C'est reproductible mais non optimal pour estimer la variabilité de sélection. Une étude doctorale de confirmation devrait créer des folds externes et plusieurs **rotations internes** de validation, sans jamais tester plusieurs fois des modèles sélectionnés sur le même fold externe. Les résultats obtenus avec 3 cellules ne permettent pas une inférence statistique forte.

### Phase 4 — Confirmer ou rejeter l'ajout de GWO

GWO n'est **pas** une source automatique d'originalité. Une étude complémentaire doit définir un espace compact, un budget équivalent entre GWO et recherche aléatoire, une fonction objectif mesurée **uniquement en validation interne**, des graines multiples et un coût CPU/GPU. Si GWO n'apporte pas une amélioration robuste, il faut l'abandonner et le justifier. Aucun script GWO n'est ajouté dans cette livraison afin de ne pas simuler une contribution non démontrée.

### Phase 5 — Limites et généralisation externe

- Variations de chimie, température, courant, tension de coupure : NASA ne suffit pas à prouver une robustesse industrielle.
- Évaluer, si disponible, un autre jeu (CALCE, Oxford, MATR) avec définition EOL harmonisée et protocoles compatibles.
- Une cellule censurée à droite nécessite un modèle de survie/traitement de censure, pas une étiquette de RUL forcée.
- Pour un usage réel, établir intervalles de prédiction, calibration, contrôle de domaine, gestion des mesures incomplètes et validation sur matériel BMS.
- Ne pas mélanger prédiction **SOH à k+1**, estimation **SOH à k**, extrapolation **trajectoire complète** et prévision **RUL à k** dans les tableaux comparatifs.

## Traçabilité des décisions et statut

| Décision | Statut | Motif |
|---|---|---|
| NASA B0005/6/7/18, `discharge` | Implémenté | Correspond aux sources du projet |
| EOL à 80 % (1,6 Ah) | Implémenté | Seuil commun proposé par les notes, à vérifier sur les .mat |
| Entrée V/I/T, 128 points + durée + rang | Implémenté | Représentation vérifiable après décharge complète |
| SOH(k+1) + RUL(k) multitâche | Implémenté | Cibles non triviales et alignées temporellement |
| Z-score TRAIN only | Implémenté et testé | Pas de fuite par standardisation |
| Batterie test indépendante | Implémenté et testé | Généralisation hors cellule |
| CNN/MHA/TCN/2 têtes | Implémenté et testé techniquement | Contribution prédictive encore à démontrer |
| GWO | Non implémenté | Absence de justification par baselines |
| Prédiction certifiée sur NASA | **Non établie** | Fichiers .mat absents de l'archive fournie |
| Performance sur batterie BMS réelle | **Non établie** | Données de banc NASA uniquement |

## Sources scientifiques principales

1. Saha et Goebel, NASA Battery Data Set (2007) : https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
2. Vaswani et al., Attention Is All You Need (2017) : https://arxiv.org/abs/1706.03762
3. Bai, Kolter, Koltun, Temporal Convolutional Networks (2018) : https://arxiv.org/abs/1803.01271
4. Zhang, Robustness of CNN-augmented sequential models for Li-ion battery RUL prediction under data scarcity (PLOS ONE, 2025) : https://doi.org/10.1371/journal.pone.0339528
5. Lithium-ion battery RUL prediction based on optimized VMD-SSA-PatchTST algorithm (Scientific Reports, 2025) : https://www.nature.com/articles/s41598-025-11934-7
6. Zhu et al., Leakage-Resistant and Censor-Aware Evaluation ... (Batteries, 2026) : https://doi.org/10.3390/batteries12110484

**Avertissement bibliographique :** les protocoles publiés diffèrent par leurs entrées, horizons, seuils EOL et splits. Les chiffres d'articles ne peuvent pas être comparés directement avec les valeurs du projet tant que les expériences ne sont pas harmonisées.
