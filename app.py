"""
Application Web Streamlit - EuroMillions Pro IA
"""

import os
import streamlit as st
import pandas as pd
from model.euromillions_pro_pipeline import run_pipeline
from model.data_manager import DataManager

# Configuration de la page
st.set_page_config(
    page_title="EuroMillions Pro IA",
    page_icon="🎰",
    layout="wide"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stTable {
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎰 Tableau de Bord & Prédictions Officielles - EuroMillions Pro IA")
st.markdown("---")

# Barre latérale pour les configurations
st.sidebar.header("⚙️ Paramètres")

# Chemin direct configuré en dur
csv_file = r"C:\Users\Kevin\Desktop\Mes Projets\Euromillions-ML\euromillions.csv"
st.sidebar.text(f"Fichier cible :\n{os.path.basename(csv_file)}")

output_dir = st.sidebar.text_input("Dossier de sortie", "output")

# Curseur pour choisir le nombre de grilles à générer
nb_grilles = st.sidebar.slider("Nombre de grilles à générer", min_value=1, max_value=20, value=6, step=1)

class Config:
    def __init__(self, n_tickets):
        self.pool_numbers = 50
        self.pool_stars = 12
        self.n_tickets = n_tickets
        self.num_tickets = n_tickets

if st.sidebar.button("🚀 Lancer l'Analyse & Générer les Grilles", type="primary"):
    if not os.path.exists(csv_file):
        st.error(f"❌ Le fichier est introuvable à l'emplacement : `{csv_file}`. Veuillez vérifier son nom exact.")
    else:
        # 1. Synchronisation automatique du CSV avec le site web
        with st.spinner("🔄 Synchronisation des derniers tirages en ligne..."):
            try:
                dm = DataManager(csv_path=csv_file)
                sync_result = dm.synchronize_database()
                if sync_result["status"] == "success":
                    st.toast(f"✅ CSV mis à jour : {sync_result['added']} nouveau(x) tirage(s) ajouté(s) !", icon="🎉")
                elif sync_result["status"] == "up_to_date":
                    st.toast("ℹ️ Le fichier CSV est déjà à jour.", icon="📌")
            except Exception as sync_err:
                st.warning(f"⚠️ Avertissement synchronisation : {sync_err}")

        # 2. Entraînement et génération
        with st.spinner("🔄 Entraînement des modèles et optimisation Monte Carlo en cours..."):
            try:
                results = run_pipeline(csv_file, output_dir, Config(nb_grilles))
                st.success("✅ Génération des grilles réussie avec succès !")
                
                # Affichage du Top IA
                st.subheader("🔮 Top National Individuel (Recommandations IA)")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**5 Numéros Clés :**")
                    nums_html = "".join([f"<span style='background-color:#d4af37; color:black; padding:6px 12px; margin-right:8px; border-radius:6px; font-weight:bold; font-size:16px;'>{n:02d}</span>" for n in results["top5_numbers"]])
                    st.markdown(nums_html, unsafe_allow_html=True)
                    
                with col2:
                    st.markdown("**2 Étoiles Clés :**")
                    stars_html = "".join([f"<span style='background-color:#d9534f; color:white; padding:6px 12px; margin-right:8px; border-radius:6px; font-weight:bold; font-size:16px;'>★ {s:02d}</span>" for s in results["top2_stars"]])
                    st.markdown(stars_html, unsafe_allow_html=True)

                st.markdown("---")

                # Affichage du Portefeuille de Couverture Optimisé
                st.subheader(f"🟢 Portefeuille de Couverture Optimisé ({nb_grilles} grilles)")
                df_port = pd.read_csv(results["portfolio_csv"])
                
                for _, row in df_port.iterrows():
                    grid_num = int(row['ticket'])
                    n1, n2, n3, n4, n5 = int(row['n1']), int(row['n2']), int(row['n3']), int(row['n4']), int(row['n5'])
                    s1, s2 = int(row['s1']), int(row['s2'])
                    
                    col_g, col_n, col_e = st.columns([1, 4, 2])
                    
                    with col_g:
                        st.markdown(f"**Grille #{grid_num:02d}**")
                        
                    with col_n:
                        nums_str = "".join([f"<span style='background-color:#d4af37; color:black; padding:4px 10px; margin-right:6px; border-radius:6px; font-weight:bold; font-size:15px;'>{n:02d}</span>" for n in [n1, n2, n3, n4, n5]])
                        st.markdown(nums_str, unsafe_allow_html=True)
                        
                    with col_e:
                        stars_str = "".join([f"<span style='background-color:#d9534f; color:white; padding:4px 10px; margin-right:6px; border-radius:6px; font-weight:bold; font-size:15px;'>★ {s:02d}</span>" for s in [s1, s2]])
                        st.markdown(stars_str, unsafe_allow_html=True)
                
                st.markdown("---")

                # Bouton de téléchargement du CSV
                with open(results["portfolio_csv"], "rb") as f:
                    st.download_button(
                        label="📥 Télécharger le fichier CSV des grilles",
                        data=f,
                        file_name=f"portfolio_{nb_grilles}_tickets.csv",
                        mime="text/csv"
                    )

                # Graphiques Statistiques robustes
                st.markdown("---")
                st.subheader("📊 Analyses Statistiques & Tendances Historiques")
                
                try:
                    df_tirages = pd.read_csv(csv_file)
                    numeric_cols = df_tirages.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    
                    potential_nums = [c for c in df_tirages.columns if any(k in c.lower() for k in ['n1', 'n2', 'n3', 'n4', 'n5', 'boule', 'num', 'ball'])]
                    potential_stars = [c for c in df_tirages.columns if any(k in c.lower() for k in ['s1', 's2', 'etoile', 'star'])]
                    
                    num_cols = potential_nums[:5] if len(potential_nums) >= 5 else numeric_cols[:5]
                    star_cols = potential_stars[:2] if len(potential_stars) >= 2 else (numeric_cols[5:7] if len(numeric_cols) >= 7 else [])

                    col_g1, col_g2 = st.columns(2)
                    
                    with col_g1:
                        st.markdown("**Fréquence d'apparition des Numéros**")
                        if num_cols:
                            all_nums = pd.concat([pd.to_numeric(df_tirages[col], errors='coerce') for col in num_cols]).dropna()
                            st.bar_chart(all_nums.value_counts().sort_index())
                        else:
                            st.info("Colonnes de numéros introuvables.")

                    with col_g2:
                        st.markdown("**Fréquence d'apparition des Étoiles**")
                        if star_cols:
                            all_stars = pd.concat([pd.to_numeric(df_tirages[col], errors='coerce') for col in star_cols]).dropna()
                            st.bar_chart(all_stars.value_counts().sort_index())
                        else:
                            st.info("Colonnes d'étoiles introuvables.")
                except Exception as chart_err:
                    st.warning(f"Impossible d'afficher les graphiques : {chart_err}")

            except Exception as e:
                st.error(f"Une erreur est survenue lors de l'exécution : {e}")
else:
    st.info("👈 Ajustez le curseur dans la barre latérale et cliquez sur le bouton pour lancer l'analyse.")