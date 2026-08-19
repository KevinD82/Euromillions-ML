# EuroMillions ML

Application Python qui analyse l'historique EuroMillions et produit des recommandations experimentales a l'aide de modeles de ranking LightGBM, CatBoost et XGBoost.

> Les tirages de loterie restent aleatoires. Les scores, le backtest et les grilles generees ne constituent pas une prediction garantie ni une methode pour augmenter les probabilites mathematiques de gain.

## Fonctionnement

Le pipeline:

1. charge et normalise le fichier historique CSV;
2. construit des caracteristiques par numero et par etoile (frequence, retard, historique);
3. entraine un ensemble de trois rankers;
4. calcule les 5 numeros et les 2 etoiles les mieux scores;
5. genere 200 grilles candidates ponderees par les scores IA;
6. selectionne 5 grilles diversifiees avec une strategie de couverture;
7. construit une 6e grille **combinee** a partir des colonnes des 5 grilles retenues.

### Grille combinee

La grille combinee reprend:

- le meilleur numero de chaque colonne `n1` a `n5` des 5 grilles IA;
- la meilleure etoile de chaque colonne `s1` et `s2`;
- le score IA pour choisir le meilleur candidat de chaque colonne;
- une resolution des doublons afin de conserver des numeros et etoiles uniques.

Elle est ajoutee en derniere position dans le portefeuille. Le nombre de grilles est fixe: il n'y a donc plus de selection dans l'interface.

## Structure

```text
.
├── app.py                              # Interface Streamlit
├── run.py                              # Lancement du mode terminal
├── main_controller.py                  # Orchestration MVC
├── config.py                           # Parametres partages
├── continuous_learning.py              # Evaluation des predictions passees
├── model/
│   ├── data_manager.py                 # Synchronisation des donnees
│   └── euromillions_pro_pipeline.py    # Chargement, entrainement et portefeuille
├── view/terminal_view.py               # Affichage Rich dans le terminal
├── euromillions.csv                    # Historique local
└── output/                             # Fichiers generes par Streamlit
```

## Installation

Python 3.10 ou plus recent est recommande.

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Les bibliotheques principales sont LightGBM, CatBoost, XGBoost, pandas, NumPy, Streamlit et Rich. Si l'environnement ne possede pas CatBoost, XGBoost ou Streamlit, installez-les avec:

```bash
pip install catboost xgboost streamlit
```

## Utilisation

### Interface web

```bash
streamlit run app.py
```

L'interface synchronise le CSV, execute le backtest, entraine les modeles et affiche:

- la recommandation IA principale (5 numeros et 2 etoiles);
- 5 grilles IA diversifiees;
- 1 grille combinee;
- les frequences historiques des numeros et des etoiles;
- un bouton de telechargement du portefeuille.

### Mode terminal

```bash
python run.py
```

Ce mode utilise le controleur MVC, l'apprentissage continu et exporte le portefeuille dans `resultats_euromillions/portfolio_6_tickets.csv`.

## Entrees et sorties

Le fichier historique doit contenir une date, 5 numeros et 2 etoiles. Le chargeur reconnait notamment:

- date: `date`, `date_de_tirage`, `Date`, `date_tirage`;
- numeros: `n1` a `n5`, `boule_1` a `boule_5`, `num1` a `num5`;
- etoiles: `s1`, `s2`, `etoile_1`, `etoile_2`, `star_1`, `star_2`.

Les colonnes numeriques sont aussi detectees automatiquement lorsque les noms ne correspondent pas a ces formats.

Le CSV produit contient toujours:

```text
ticket,n1,n2,n3,n4,n5,s1,s2
1,...
2,...
3,...
4,...
5,...
6,...   # grille combinee
```

## Validation et limites

Le pipeline lance un backtest temporel sur les derniers tirages disponibles et calcule un bilan financier indicatif selon une grille de gains configuree dans le code. Ce bilan sert a comparer des strategies historiques; il ne predit pas les performances futures.

Les scores IA mesurent des tendances presentes dans l'historique. Ils ne transforment pas un tirage aleatoire en evenement predictible. Jouez de maniere responsable.

## Developpement

Verifier la syntaxe des modules principaux:

```bash
python -m py_compile app.py config.py main_controller.py model\euromillions_pro_pipeline.py view\terminal_view.py
```

Les fichiers volumineux generes par les entrainements (`cache/`, `catboost_info/` et les sorties CSV) sont des artefacts locaux et ne sont pas necessaires pour comprendre le code source.
