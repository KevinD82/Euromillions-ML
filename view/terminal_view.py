#!/usr/bin/env python3
"""
Vue (MVC) - Affichages et tableaux de bord dans le terminal avec Rich
Version Premium & Ultra-Design
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
        """Titre principal stylisé en mode Dashboard Pro."""
        self.console.print("\n")
        header_text = (
            "[bold bright_cyan]⚡ EUROMILLIONS PRO IA ⚡[/bold bright_cyan]\n"
            "[dim italic]Architecture MVC • Pipeline Prédictif LightGBM & Deep Learning[/dim italic]"
        )
        self.console.print(
            Panel.fit(
                Align.center(header_text),
                border_style="bright_magenta",
                padding=(1, 4),
            )
        )
        self.console.print()

    def display_status(self, message: str, status_type: str = "info"):
        """Affiche une notification colorée et épurée dans la console."""
        colors = {
            "info": "bright_cyan",
            "success": "bright_green",
            "warning": "bright_yellow",
            "error": "bright_red",
        }
        prefixes = {
            "info": "🔍",
            "success": "✨",
            "warning": "⚠️",
            "error": "❌",
        }
        self.console.print(
            f"[{colors.get(status_type, 'white')}] {prefixes.get(status_type, '•')} {message}[/]"
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
        """Génère le tableau de bord final hautement stylisé."""
        self.console.print("\n")

        # En-tête de section principal
        self.console.print(
            Panel.fit(
                "[bold bright_white]📊 TABLEAU DE BORD & PRÉDICTIONS OFFICIELLES[/bold bright_white]",
                border_style="bright_blue",
                padding=(0, 2),
            )
        )
        self.console.print()

        # 1. Tableau des Favoris Absolus (Design Cartes)
        table_top = Table(
            title="🔮 [bold bright_magenta]Top National Individuel (Recommandations IA)[/bold bright_magenta]",
            title_justify="left",
            border_style="dim",
            box=None,  # Style épuré sans grille lourde
            pad_edge=False,
        )
        table_top.add_column("Type", style="bold cyan", width=22)
        table_top.add_column("Sélection Optimisée", justify="left")

        # Boules dorées stylisées
        str_nums = "   ".join(
            f"[bold black on bright_yellow]  {int(n):02d}  [/]" for n in top_numbers
        )
        # Étoiles rouges stylisées
        str_stars = "   ".join(
            f"[bold white on bright_red]  ★ {int(s):02d}  [/]" for s in top_stars
        )

        table_top.add_row("5 Numéros Clés", str_nums)
        table_top.add_row("", "")  # Espace vide
        table_top.add_row("2 Étoiles Clés", str_stars)

        self.console.print(
            Panel(
                table_top,
                border_style="magenta",
                padding=(1, 2),
                title_align="left",
            )
        )
        self.console.print()

        # 2. Tableau du Portefeuille de tickets (Design moderne avec bordures arrondies)
        path_obj = Path(portfolio_path)
        if path_obj.exists():
            df_port = pd.read_csv(path_obj)

            table_port = Table(
                title="🎟️ [bold bright_green]Portefeuille: 5 grilles IA + 1 grille combinée[/bold bright_green]",
                title_justify="left",
                border_style="bright_green",
                header_style="bold bright_white on dark_green",
                expand=True,
            )
            table_port.add_column(
                "Grille", justify="center", style="bold dim cyan", width=10
            )
            table_port.add_column(
                "Numéros Sélectionnés (1 à 50)", justify="left", style="bold white"
            )
            table_port.add_column("Étoiles", justify="center", width=18)

            for _, row in df_port.iterrows():
                nums_list = sorted([int(row[f"n{i}"]) for i in range(1, 6)])
                stars_list = sorted([int(row[f"s{i}"]) for i in range(1, 3)])

                # Affichage propre des numéros avec pastilles discrètes
                nums_str = "   ".join(
                    f"[bright_cyan]{n:02d}[/bright_cyan]" for n in nums_list
                )
                stars_str = f"[bright_red]★ {stars_list[0]:02d}[/bright_red]   [bright_red]★ {stars_list[1]:02d}[/bright_red]"

                ticket_id = int(row["ticket"])
                label = "Combinée" if ticket_id == 6 else f"IA #{ticket_id:02d}"
                table_port.add_row(label, nums_str, stars_str)

            self.console.print(table_port)
            self.console.print()

            # Encadré discret pour le chemin du fichier CSV
            self.console.print(
                Panel(
                    f"[dim]💾 Fichier exporté avec succès :[/dim] [bold underline cyan]{portfolio_path}[/bold underline cyan]",
                    border_style="dim",
                    padding=(0, 1),
                )
            )
            self.console.print()
        else:
            self.display_status(
                "Le fichier du portefeuille de grilles est introuvable.",
                "warning",
            )
