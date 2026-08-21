"""
Contrôleur (MVC) - Orchestrateur EuroMillions avec Apprentissage Continu
"""

import pandas as pd

from config import Config
from continuous_learning import ContinuousLearningEngine
from model.euromillions_pro_pipeline import (
    build_long_table,
    load_draws,
    run_pipeline,
)
from view.terminal_view import TerminalView


class MainController:
    def __init__(self):
        """Initialise le Modèle, la Vue et la Configuration."""
        self.cfg = Config()
        self.cl_engine = ContinuousLearningEngine()
        self.view = TerminalView()

    def run(self):
        """Exécute le pipeline complet du programme."""
        self.view.display_header()
        self.view.display_status("Chargement et synchronisation des données...", "info")

        df_draws = load_draws("euromillions.csv")

        # 1. ÉVALUATION AUTOMATIQUE DU TIRAGE DE LA SEMAINE
        self.view.display_status("Vérification des prédictions passées...", "info")
        self.cl_engine.evaluate_past_predictions(df_draws)

        # 2. CONTRÔLE DE LA DÉRIVE DE CONCEPT (Drift Tracking)
        is_drifted = self.cl_engine.check_concept_drift(threshold_ndcg=0.35)

        if is_drifted:
            self.view.display_status(
                "Dérive détectée. Ré-optimisation des hyperparamètres en cours...",
                "warning",
            )
            long_num = build_long_table(
                df_draws, self.cfg.pool_numbers, "number", self.cfg
            )
            X = long_num.drop(columns=["draw_idx", "date", "entity_id", "label"])
            y = long_num["label"].values

            new_params = self.cl_engine.run_optuna_retuning(X, y, kind="numbers")
            if new_params:
                self.cfg.lgb_classifier_params_numbers.update(new_params)

        # 3. APPRENTISSAGE INCRÉMENTAL (Warm Start) ET GÉNÉRATION
        self.view.display_status("Exécution du pipeline prédictif...", "info")
        results = run_pipeline("euromillions.csv", "resultats_euromillions", self.cfg)

        # 4. JOURNALISATION DE LA NOUVELLE PRÉDICTION
        next_date = (df_draws["date"].max() + pd.Timedelta(days=3)).strftime("%Y-%m-%d")

        portfolio_data = []
        df_port = pd.read_csv(results["portfolio_csv"])
        for _, row in df_port.iterrows():
            portfolio_data.append({
                "nums": [int(row[f"n{k}"]) for k in range(1, 6)],
                "stars": [int(row[f"s{k}"]) for k in range(1, 3)],
            })

        self.cl_engine.log_prediction(
            target_draw_date=next_date,
            top_nums=results["top5_numbers"],
            top_stars=results["top2_stars"],
            portfolio=portfolio_data,
        )

        # ==========================================
        # 5. AFFICHAGE DU TABLEAU DE BORD (Via la Vue MVC)
        # ==========================================
        self.view.display_dashboard(
            top_numbers=results["top5_numbers"],
            top_stars=results["top2_stars"],
            portfolio_path=results["portfolio_csv"],
        )

        self.view.display_status(
            "Processus d'apprentissage continu terminé avec succès !", "success"
        )
