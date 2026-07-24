#!/usr/bin/env python3
"""
Orchestrateur Principal (Contrôleur - MVC) - Menu Interactif ML Optimisé
"""

import sys

from rich.console import Console
from rich.panel import Panel

try:
    from model.data_manager import DataManager
except ImportError:
    print("❌ Erreur : Impossible de charger 'model/data_manager.py'.")
    sys.exit(1)

console = Console()


def afficher_menu() -> None:
    """Affiche un joli menu interactif dans la console."""
    console.print(
        "\n[bold cyan]==================================================[/bold cyan]"
    )
    console.print(
        "[bold white] 🎰 ORCHESTRATEUR DE PRÉDICTION EUROMILLIONS 🎰[/bold white]"
    )
    console.print(
        "[bold cyan]==================================================[/bold cyan]\n"
    )

    console.print(
        Panel(
            "[bold green]1. Lancer le Pipeline d'IA (LightGBM Optimizer)[/bold green]\n\n"
            "[white]• [bold]Comment ça marche :[/bold] Synchronise la base de données, entraîne des arbres de décision "
            "boostés et applique un calibrage statistique (Isotone).\n"
            "• [bold]Points forts :[/bold] Ultra-rapide, trie par pertinence et applique des filtres physiques d'étalement.\n"
            "• [bold]Résultat :[/bold] Génère un portefeuille diversifié de 7 grilles réalistes uniques.[/white]",
            title="🤖 MOTEUR MACHINE LEARNING",
            expand=False,
        )
    )

    console.print(Panel("[bold red]2. Quitter le programme[/bold red]", expand=False))


def main():
    # 1. Init de la BDD et Synchronisation automatique au démarrage
    db = DataManager()

    console.print(
        "[bold cyan]🔄 Synchronisation automatique avec Les Bons Numéros...[/bold cyan]"
    )
    try:
        sync_info = db.synchronize_database()
        if sync_info["status"] == "success":
            console.print(
                f"[bold green]✅ Base synchronisée ! "
                f"+{sync_info['added']} tirages ajoutés. Total : {sync_info['total']}.[/bold green]"
            )
        else:
            console.print(
                f"[bold blue]ℹ️ Base locale à jour ({sync_info['total']} tirages). "
                f"Dernière date : {sync_info['last_date']}[/bold blue]"
            )
    except Exception as e:  # noqa: BLE001
        console.print(
            f"[bold red]⚠️ Échec de la synchronisation automatique : {e}[/bold red]"
        )
        console.print(
            "[yellow]Le programme continue avec les données locales actuelles.[/yellow]"
        )

    # 2. Boucle du menu interactif
    while True:
        afficher_menu()
        try:
            choix = input("👉 Entrez votre choix (1 ou 2) : ").strip()
        except KeyboardInterrupt:
            console.print(
                "\n[bold red]⛔ Programme interrompu par l'utilisateur.[/bold red]"
            )
            sys.exit(0)

        if choix == "1":
            console.print(
                "\n[bold green]🚀 Lancement du pipeline LightGBM et calculs prédictifs...[/bold green]\n"
            )
            try:
                from main_controller import MainController

                controller = MainController()
                controller.run_application()
            except Exception as e:  # noqa: BLE001
                console.print(
                    f"[bold red]❌ Erreur lors de l'exécution du contrôleur : {e}[/bold red]"
                )
            break

        elif choix == "2":
            console.print(
                "\n[bold yellow]👋 Fermeture du programme. Bonne chance ![/bold yellow]\n"
            )
            break
        else:
            console.print(
                "[bold red]❌ Choix invalide, veuillez entrer 1 ou 2.[/bold red]"
            )


if __name__ == "__main__":
    main()
