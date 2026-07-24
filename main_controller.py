#!/usr/bin/env python3
"""
Contrôleur (MVC) - Orchestration du pipeline de données et d'IA (LightGBM)
Version épurée Pure ML
"""
import sys
import os
import pandas as pd

# Sécurité pour s'assurer que le dossier racine est dans le PYTHONPATH
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from model.data_manager import DataManager
from view.terminal_view import TerminalView
import view.euromillions_pro_pipeline as euromillions_pro_pipeline

class MainController:
    def __init__(self):
        self.model = DataManager()
        self.view = TerminalView()
        self.output_dir = "resultats_euromillions"

    def run_application(self):
        """Exécute l'application de bout en bout (pipeline ML)."""
        self.view.display_header()
        
        # Le rafraîchissement initial et la synchronisation étant gérés dans run.py,
        # nous lançons immédiatement l'entraînement et l'analyse prédictive.
        self.view.display_status("Exécution du pipeline d'IA (LightGBM Optimizer) sur l'historique global...", "info")
        
        # Simulation d'arguments CLI pour alimenter proprement le pipeline d'IA sans os.system
        original_argv = sys.argv.copy()
        sys.argv = [
            "euromillions_pro_pipeline.py",
            "--csv", self.model.csv_path,
            "--out", self.output_dir
        ]
        
        try:
            # Appel direct de la fonction principale du pipeline ML
            euromillions_pro_pipeline.main()
            self.view.display_status("Calculs prédictifs, backtesting et filtrage géométrique terminés avec succès.", "success")
            print("-" * 60)
        except Exception as e:
            self.view.display_status(f"Erreur critique dans le moteur d'IA : {e}", "error")
            return
        finally:
            # Restauration systématique des arguments initiaux du système
            sys.argv = original_argv

        portfolio_csv = os.path.join(self.output_dir, "portfolio_7_tickets.csv")
        
        # Récupération et chargement des résultats générés par LightGBM pour l'affichage final
        try:
            df_nums = pd.read_csv(os.path.join(self.output_dir, "pred_numbers_next.csv"))
            df_stars = pd.read_csv(os.path.join(self.output_dir, "pred_stars_next.csv"))
            
            # Extraction des favoris absolus (Top des scores calibrés par l'Isotone Regression)
            top5_nums = df_nums.sort_values(by="score_fused", ascending=False).head(5)["entity_id"].tolist()
            top2_stars = df_stars.sort_values(by="score_fused", ascending=False).head(2)["entity_id"].tolist()
            
            # Transmission à la vue pour affichage du tableau de bord Rich
            self.view.display_dashboard(top5_nums, top2_stars, portfolio_csv)
            
        except Exception as e:
            self.view.display_status(f"Erreur lors de la récupération des résultats pour affichage : {e}", "error")