from dataclasses import dataclass


@dataclass
class Config:
    # --- Pool ---
    pool_numbers: int = 50
    pool_stars: int = 12

    # --- History ---
    min_history_draws: int = 300
    allow_older_regimes: bool = False

    # --- Backtest ---
    n_splits_backtest: int = 20  # ➡️ Ajouté ici pour un pilotage centralisé

    # --- Cooccurrence / decay ---
    half_life_numbers: float = 60.0
    half_life_stars: float = 40.0
    cooc_half_life_numbers: float = 40.0
    cooc_half_life_stars: float = 30.0
    cooc_topk: int = 10

    # --- Windows (réduites pour perf) ---
    windows_numbers: tuple = (10, 25, 50)
    windows_stars: tuple = (10, 25, 50)

    # --- OOF splits (Optimisés à 3 pour diviser le temps par deux) ---
    oof_splits_numbers: int = 3
    oof_splits_stars: int = 3

    # --- Eval ---
    eval_at_numbers: int = 5
    eval_at_stars: int = 2

    # --- Échantillonnage & Portefeuille ---
    sampling_alpha_numbers: float = 0.5
    sampling_alpha_stars: float = 0.5
    portfolio_max_attempts: int = 5000

    # --- LightGBM ranker (ML optimisé CPU) ---
    lgb_ranker_params_numbers: dict = None
    lgb_ranker_params_stars: dict = None

    # --- LightGBM classifier (ML optimisé CPU) ---
    lgb_classifier_params_numbers: dict = None
    lgb_classifier_params_stars: dict = None

    # --- Fusion ---
    ranker_weight_numbers: float = 0.6
    classifier_weight_numbers: float = 0.4
    ranker_weight_stars: float = 0.6
    classifier_weight_stars: float = 0.4

    def __post_init__(self):
        # ----------------------------------------------------------------------
        # CONFIGURATION DES PARAMÈTRES LIGHTGBM (Optimisation Pure CPU Parallèle)
        # ----------------------------------------------------------------------

        # Ranker (Ordonnancement des numéros)
        self.lgb_ranker_params_numbers = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "learning_rate": 0.05,
            "num_leaves": 32,
            "n_estimators": 200,
            "min_data_in_leaf": 5,  # ✅ Protection contre le "Abort" à 40%
            "feature_fraction": 1.0,  # ✅ Stabilité d'analyse globale
            "bagging_fraction": 1.0,  # ✅ Évite les sous-échantillons vides
            "bagging_freq": 0,  # ✅ Désactivation du bagging pour les petits splits
            "verbosity": -1,
            "device_type": "cpu",  # Forçage sur CPU natif
            "n_jobs": -1,  # Parallélisation automatique sur tous les cœurs du CPU
        }

        self.lgb_ranker_params_stars = self.lgb_ranker_params_numbers.copy()

        # Classifier (Probabilité binaire de sortie)
        self.lgb_classifier_params_numbers = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 32,
            "n_estimators": 200,
            "min_data_in_leaf": 5,  # ✅ Protection contre le "Abort"
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "bagging_freq": 0,
            "verbosity": -1,
            "device_type": "cpu",
            "n_jobs": -1,
        }

        self.lgb_classifier_params_stars = self.lgb_classifier_params_numbers.copy()
        self.lgb_classifier_params_stars["num_leaves"] = 16
