#!/usr/bin/env python3
"""
Point d'entrée principal - Architecture MVC EuroMillions
"""

import os
import sys

# Ajoute le dossier racine du projet au chemin de recherche Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_controller import MainController


def main():
    # Instanciation et lancement du contrôleur principal
    controller = MainController()
    controller.run()


if __name__ == "__main__":
    main()
