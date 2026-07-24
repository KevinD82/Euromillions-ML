#!/usr/bin/env python3
"""
Vue (MVC) - Affichages et tableaux de bord dans le terminal avec Rich
Version épurée Pure ML - Entièrement Corrigée
"""

from pathlib import Path

import pandas as pd
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


class TerminalView:
    def __init__(self):
        self.console = Console()

    def display_header(self):
        """Titre principal stylisé."""
        self.console.print("\n")
        header_text = (
            "[bold cyan]EUROMILLIONS PRO IA[/bold cyan] 🎰\n"
            "[dim]Architecture MVC & Pipeline Prédictif LightGBM Optimisé[/dim]"
        )
        self.console.print(Panel.fit(Align.center(header_text), border_style="purple"))

    def display_status(self, message: str, status_type: str = "info"):
        """Affiche une notification colorée dans la console."""
        colors = {
            "info": "cyan",
            "success": "green",
            "warning": "yellow",
            "error": "red",
        }
        prefixes = {"info": "💡", "success": "✅", "warning": "⚠️", "error": "❌"}
        self.console.print(
            f"[{colors.get(status_type, 'white')}]{prefixes.get(status_type, '•')} {message}[/]"
        )

    def display_sync_report(self, report: dict):
        """Affiche le bilan détaillé de la synchronisation des données."""
        if report["status"] == "success":
            self.display_status("Base de données mise à jour avec succès !", "success")
            self.console.print(
                f"   [dim]• Nouveaux tirages ajoutés :[/dim] [bold green]+{report['added']}[/]"
            )
        elif report["status"] == "up_to_date":
            self.display_status("La base locale est déjà parfaitement à jour.", "info")

        self.console.print(
            f"   [dim]• Nombre total de tirages en base :[/dim] [bold white]{report['total']}[/]"
        )
        self.console.print(
            f"   [dim]• Dernier tirage enregistré :[/dim] [bold yellow]{report['last_date']}[/]\n"
        )

    def display_dashboard(
        self, top_numbers: list, top_stars: list, portfolio_path: str
    ):
        """Génère le tableau de bord final contenant les prédictions et les grilles filtrées."""
        self.console.print("\n")
        self.console.print(
            "[bold underline white]📊 TABLEAU DE BORD DES PRÉDICTIONS DE L'IA[/bold underline white]\n"
        )

        # 1. Tableau des Favoris Absolus
        table_top = Table(
            title="🔮 Statistique Individuelle (Top National)",
            title_style="bold magenta",
        )
        table_top.add_column("Type Element", justify="center", style="bold white")
        table_top.add_column("Sélection de l'IA", justify="left")

        str_nums = "  ".join(
            f"[bold black on yellow] {int(n)} [/]" for n in top_numbers
        )
        str_stars = "  ".join(f"[bold white on red] ★ {int(s)} [/]" for s in top_stars)

        table_top.add_row("5 Numéros Favoris", str_nums)
        table_top.add_row("2 Étoiles Favorites", str_stars)
        self.console.print(table_top)
        self.console.print("\n")

        # 2. Tableau du Portefeuille de tickets (couverture équilibrée par paliers)
        path_obj = Path(portfolio_path)
        if path_obj.exists():
            df_port = pd.read_csv(path_obj)
            table_port = Table(
                title=(
                    "🎟️ Portefeuille de Couverture Optimisé (Filtre Géométrique Actif)"
                ),
                title_style="bold green",
            )
            table_port.add_column("Grille", justify="center", style="dim")
            table_port.add_column(
                "Numéros Sélectionnés (Étalés par Dizaines)", justify="left"
            )
            table_port.add_column("Étoiles", justify="center")

            for _, row in df_port.iterrows():
                # Re-tri local des numéros pour un affichage visuel impeccable de 1 à 50
                nums_list = sorted([int(row[f"n{i}"]) for i in range(1, 6)])
                stars_list = sorted([int(row[f"s{i}"]) for i in range(1, 3)])

                nums_str = ", ".join(f"[cyan]{n}[/cyan]" for n in nums_list)
                stars_str = (
                    f"[red]★ {stars_list[0]}[/red] - [red]★ {stars_list[1]}[/red]"
                )

                table_port.add_row(
                    f"Ticket {int(row['ticket']):02d}", nums_str, stars_str
                )

            self.console.print(table_port)
            self.console.print(
                f"\n[dim]💾 Portefeuille complet sauvegardé sous : {portfolio_path}[/dim]\n"
            )
        else:
            self.display_status(
                "Le fichier du portefeuille de grilles est introuvable.",
                "warning",
            )
