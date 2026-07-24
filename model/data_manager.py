#!/usr/bin/env python3
"""
Modèle (MVC) - Gestion de la base de données et scraping ciblé depuis Les Bons Numéros
"""
import os
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup

class DataManager:
    def __init__(self, csv_path: str = "euromillions.csv"):
        self.csv_path = csv_path

    def _parse_french_date(self, text_date: str) -> str:
        """Convertit une date textuelle française (ex: 'Mardi 23 Juin') en format 'JJ/MM/AAAA'."""
        months_fr = {
            'janvier': '01', 'février': '02', 'mars': '03', 'avril': '04', 'mai': '05', 'juin': '06',
            'juillet': '07', 'août': '08', 'septembre': '09', 'octobre': '10', 'novembre': '11', 'décembre': '12'
        }
        try:
            text_cleaned = text_date.lower().replace('&nbsp;', ' ').replace('\xa0', ' ').strip()
            # On cherche le jour (1 ou 2 chiffres) et le mois en toutes lettres
            match = re.search(r"(\d{1,2})\s+([a-zéeûâoû]+)", text_cleaned)
            if match:
                day = f"{int(match.group(1)):02d}"
                month_str = match.group(2)
                month = months_fr.get(month_str)
                
                # Comme la page n'affiche pas toujours l'année dans le libellé court, 
                # on utilise 2026 par défaut (ou l'année courante)
                year = "2026"
                
                if month:
                    return f"{day}/{month}/{year}"
        except Exception:
            pass
        return None

    def fetch_all_visible_draws(self) -> list:
        """Scrape les tirages en ciblant précisément les classes 'row stripped' du site."""
        url = "https://www.lesbonsnumeros.com/euromillions/resultats/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        scraped_draws = []
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Ciblage direct des lignes de tirage visibles dans ton extrait de code
            rows = soup.find_all('div', class_=lambda c: c and 'row stripped' in c)
            
            for row in rows:
                # 1. Extraction de la date depuis le lien <a>
                link_el = row.find('a')
                if not link_el:
                    continue
                
                # On nettoie le texte du lien (ex: "Résultat Euromillions Mardi 23 Juin" -> "23 Juin")
                raw_link_text = link_el.get_text()
                block_date = self._parse_french_date(raw_link_text)
                
                if not block_date:
                    continue

                # 2. Extraction des boules numéros
                balls = []
                for li in row.find_all('li', class_='numero'):
                    v = li.get_text().strip()
                    if v.isdigit():
                        balls.append(int(v))
                        
                # 3. Extraction des boules étoiles
                stars = []
                for li in row.find_all('li', class_='etoile'):
                    v = li.get_text().strip()
                    if v.isdigit():
                        stars.append(int(v))

                # Validation stricte du tirage avant de l'ajouter
                if len(balls) == 5 and len(stars) == 2:
                    balls.sort()
                    stars.sort()
                    
                    if not any(d['date_de_tirage'] == block_date for d in scraped_draws):
                        scraped_draws.append({
                            'date_de_tirage': block_date,
                            'boule_1': balls[0], 'boule_2': balls[1], 'boule_3': balls[2], 'boule_4': balls[3], 'boule_5': balls[4],
                            'etoile_1': stars[0], 'etoile_2': stars[1],
                            'fichier_source': 'lesbonsnumeros_live'
                        })
                        
        except Exception as e:
            print(f"❌ Erreur lors du scraping ciblé : {e}")
            
        return scraped_draws

    def synchronize_database(self) -> dict:
        """Nettoie la base locale et applique le balayage chronologique montant."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Base locale '{self.csv_path}' introuvable.")
            
        df_local = pd.read_csv(self.csv_path, sep=",")
        df_local = df_local.dropna(subset=['date_de_tirage'])
        
        # NETTOYAGE : Suppression radicale des précédentes erreurs s'il y en a
        if not df_local.empty:
            df_local['date_de_tirage'] = df_local['date_de_tirage'].astype(str).str.strip()
            df_local = df_local[~df_local['date_de_tirage'].str.contains('23/06/2026', na=False)]
            df_local = df_local[df_local['boule_1'] <= 50]
        
        # Récupération des tirages depuis l'approche ciblée
        online_draws = self.fetch_all_visible_draws()
        
        if not online_draws:
            return {
                "status": "error",
                "message": "Aucun tirage n'a pu être extrait avec la nouvelle approche.",
                "added": 0,
                "total": len(df_local),
                "last_date": df_local['date_de_tirage'].iloc[-1] if not df_local.empty else "Aucune"
            }
            
        # Détection des nouveautés manquantes dans le CSV
        local_dates = set(df_local['date_de_tirage'].values)
        new_draws_to_add = [d for d in online_draws if d['date_de_tirage'] not in local_dates]
        
        if not new_draws_to_add:
            return {
                "status": "up_to_date",
                "added": 0,
                "total": len(df_local),
                "last_date": df_local['date_de_tirage'].iloc[-1]
            }
            
        # Inversion de l'ordre pour pousser du plus ancien manquant au plus récent au bas du CSV
        df_new = pd.DataFrame(new_draws_to_add).iloc[::-1]
        
        # Fusion et sauvegarde
        df_combined = pd.concat([df_local, df_new], ignore_index=True)
        df_combined.to_csv(self.csv_path, index=False, sep=",")
        
        return {
            "status": "success",
            "added": len(df_new),
            "total": len(df_combined),
            "last_date": df_combined['date_de_tirage'].iloc[-1]
        }