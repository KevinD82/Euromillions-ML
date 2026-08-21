#!/usr/bin/env python3
"""
EuroMillions ML — Module d'Apprentissage Continu, Warm Start & Tracking
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class ContinuousLearningEngine:
    def __init__(self, db_path="data/tracker.db", models_dir="models/saved"):
        self.db_path = Path(db_path)
        self.models_dir = Path(models_dir)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self._init_db()

    # ==========================================
    # 💾 1. JOURNALISATION & TRACKING (SQLite)
    # ==========================================
    def _init_db(self):
        """Initialise la base de données de suivi des prédictions."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    predicted_at TIMESTAMP,
                    target_draw_date TEXT UNIQUE,
                    top_numbers TEXT,
                    top_stars TEXT,
                    portfolio_json TEXT,
                    actual_numbers TEXT DEFAULT NULL,
                    actual_stars TEXT DEFAULT NULL,
                    ndcg_5_numbers REAL DEFAULT NULL,
                    ndcg_2_stars REAL DEFAULT NULL,
                    matched_in_portfolio REAL DEFAULT NULL,
                    evaluated INTEGER DEFAULT 0
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS drift_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    eval_date TIMESTAMP,
                    rolling_ndcg_30d REAL,
                    rolling_ndcg_90d REAL,
                    drift_detected INTEGER
                )
            """)
            conn.commit()

    def log_prediction(
        self, target_draw_date: str, top_nums: list, top_stars: list, portfolio: list
    ):
        """Enregistre une nouvelle prédiction faite par l'IA."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO predictions_history 
                (predicted_at, target_draw_date, top_numbers, top_stars, portfolio_json)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    str(target_draw_date),
                    json.dumps(top_nums),
                    json.dumps(top_stars),
                    json.dumps(portfolio),
                ),
            )
            conn.commit()
        print(f"✅ Prédiction pour le {target_draw_date} journalisée en BDD.")

    def evaluate_past_predictions(self, df_draws: pd.DataFrame):
        """Vérifie les prédictions passées non évaluées avec les vrais tirages."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            pending = pd.read_sql_query(
                "SELECT * FROM predictions_history WHERE evaluated = 0", conn
            )

            if pending.empty:
                return

            for _, row in pending.iterrows():
                t_date = pd.to_datetime(row["target_draw_date"])
                match = df_draws[df_draws["date"] == t_date]

                if not match.empty:
                    draw = match.iloc[0]
                    act_nums = [int(draw[f"n{k}"]) for k in range(1, 6)]
                    act_stars = [int(draw[f"s{k}"]) for k in range(1, 3)]

                    top_n = json.loads(row["top_numbers"])
                    top_s = json.loads(row["top_stars"])
                    port = json.loads(row["portfolio_json"])

                    # Calcul du matching
                    match_n = len(set(top_n).intersection(set(act_nums)))
                    match_s = len(set(top_s).intersection(set(act_stars)))

                    # Taux de match portefeuille
                    port_matches = []
                    for t in port:
                        pn = len(set(t["nums"]).intersection(set(act_nums)))
                        ps = len(set(t["stars"]).intersection(set(act_stars)))
                        port_matches.append(pn + ps)

                    avg_port_match = (
                        float(np.mean(port_matches)) if port_matches else 0.0
                    )

                    cursor.execute(
                        """
                        UPDATE predictions_history 
                        SET actual_numbers = ?, actual_stars = ?, 
                            ndcg_5_numbers = ?, ndcg_2_stars = ?, 
                            matched_in_portfolio = ?, evaluated = 1
                        WHERE id = ?
                    """,
                        (
                            json.dumps(act_nums),
                            json.dumps(act_stars),
                            float(match_n / 5.0),
                            float(match_s / 2.0),
                            avg_port_match,
                            row["id"],
                        ),
                    )
            conn.commit()
            print("🎯 Évaluation des prédictions passées mise à jour.")

    # ==========================================
    # 🔄 2. APPRENTISSAGE INCRÉMENTAL (Warm Start)
    # ==========================================
    def train_or_update_lgb(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        model_key: str,
        params: dict,
        incremental: bool = True,
    ):
        """Entraîne un modèle LightGBM ou effectue une mise à jour incrémentale (Warm Start)."""
        model_file = self.models_dir / f"{model_key}.booster"

        # Nettoyage des données pour LightGBM
        cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
        for col in cat_cols:
            X[col] = X[col].astype("category")

        train_data = lgb.Dataset(X, label=y, free_raw_data=False)

        if incremental and model_file.exists():
            print(f"🔄 Apprentissage incrémental (Warm Start) pour : {model_key}")
            init_booster = lgb.Booster(model_file=str(model_file))

            # Injection du nouveau tirage dans l'ancien modèle (10 arbres supplémentaires)
            params_inc = params.copy()
            params_inc["learning_rate"] = params.get("learning_rate", 0.05) * 0.5

            booster = lgb.train(
                params_inc,
                train_data,
                num_boost_round=10,
                init_model=init_booster,
                keep_training_booster=True,
            )
        else:
            print(f"⚙️ Entraînement complet initial pour : {model_key}")
            booster = lgb.train(params, train_data, num_boost_round=100)

        booster.save_model(str(model_file))
        return booster

    # ==========================================
    # 📉 3. CONCEPT DRIFT & OPTUNA RE-TUNING
    # ==========================================
    def check_concept_drift(self, threshold_ndcg: float = 0.35) -> bool:
        """Détecte si la précision glissante sur 30 ou 90 jours passe sous un seuil critique."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM predictions_history WHERE evaluated = 1 ORDER BY target_draw_date DESC",
                conn,
            )

        if len(df) < 8:
            return False  # Pas assez de recul

        rolling_30d = df.head(8)["ndcg_5_numbers"].mean()
        rolling_90d = (
            df.head(24)["ndcg_5_numbers"].mean() if len(df) >= 24 else rolling_30d
        )

        drift_detected = bool(rolling_30d < threshold_ndcg)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO drift_metrics (eval_date, rolling_ndcg_30d, rolling_ndcg_90d, drift_detected)
                VALUES (?, ?, ?, ?)
            """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    float(rolling_30d),
                    float(rolling_90d),
                    int(drift_detected),
                ),
            )
            conn.commit()

        if drift_detected:
            print(
                f"⚠️ DÉRIVE DE CONCEPT DÉTECTÉE ! NDCG moyen (30d) = {rolling_30d:.2f} < Seuil ({threshold_ndcg})"
            )
        return drift_detected

    def run_optuna_retuning(
        self, X: pd.DataFrame, y: np.ndarray, kind: str = "number"
    ) -> dict:
        """Ré-optimise automatiquement les hyperparamètres via Optuna en cas de dérive."""
        if not OPTUNA_AVAILABLE:
            print("❌ Optuna non installé. Impossible de ré-optimiser.")
            return {}

        print(f"🛠️ Lancement du ré-alignement automatique Optuna ({kind})...")

        def objective(trial):
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.1, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "verbose": -1,
            }

            ds = lgb.Dataset(X, label=y)
            cv_res = lgb.cv(params, ds, nfold=3, stratified=True, shuffle=False)
            return cv_res["valid binary_logloss-mean"][-1]

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=15, show_progress_bar=True)

        print(f"✨ Nouveaux hyperparamètres trouvés ({kind}) :", study.best_params)
        return study.best_params
