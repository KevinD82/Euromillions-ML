"""
Application Web Streamlit - EuroMillions Pro IA
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from config import Config
from model.data_manager import DataManager
from model.euromillions_pro_pipeline import run_pipeline

st.set_page_config(
    page_title="EuroMillions IA",
    page_icon="🎰",
    layout="wide",
)

st.title("🎰 Tableau de bord EuroMillions IA")
st.warning(
    "Le modèle apprend les régularités historiques, mais aucun modèle "
    "ne peut garantir un tirage aléatoire."
)

project_dir = Path(__file__).resolve().parent
csv_file = project_dir / "euromillions.csv"
output_dir = st.sidebar.text_input("Dossier de sortie", "output")

st.sidebar.header("⚙️ Paramètres")
st.sidebar.write(f"Historique : `{csv_file.name}`")

if st.sidebar.button(
    "🚀 Entraîner le modèle et générer les grilles",
    type="primary",
):
    if not csv_file.exists():
        st.error(f"Fichier introuvable : {csv_file}")
        st.stop()

    with st.spinner("Synchronisation des tirages..."):
        try:
            manager = DataManager(str(csv_file))
            synchronization = manager.synchronize_database()

            if synchronization.get("status") == "success":
                st.success(
                    f"{synchronization.get('added', 0)} nouveau(x) tirage(s) ajouté(s)."
                )
        except Exception as error:  # noqa: BLE001
            st.warning(f"Synchronisation ignorée : {error}")

    with st.spinner("Entraînement CatBoost et backtest..."):
        try:
            results = run_pipeline(
                str(csv_file),
                output_dir,
                Config(),
            )
        except Exception as error:  # noqa: BLE001
            st.error(f"Erreur pendant le pipeline : {error}")
            st.stop()

    st.success("✅ Modèle entraîné et grilles générées.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 5 numéros appris")

        top_numbers = results["top5_numbers"]

        for position, number in enumerate(top_numbers, start=1):
            st.write(f"**{position}.** {number}")

    with col2:
        st.subheader("Top 2 étoiles apprises")

        top_stars = results["top2_stars"]

        for position, star in enumerate(top_stars, start=1):
            st.write(f"**{position}.** {star}")

    st.subheader("🎟️ Portefeuille généré")

    portfolio = pd.read_csv(results["portfolio_csv"])

    # Supprime un éventuel index enregistré dans un ancien CSV.
    unnamed_columns = [
        column for column in portfolio.columns if column.lower().startswith("unnamed:")
    ]

    if unnamed_columns:
        portfolio = portfolio.drop(columns=unnamed_columns)

    # L'index automatique Streamlit est masqué.
    st.dataframe(
        portfolio,
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "📥 Télécharger les grilles",
        data=Path(results["portfolio_csv"]).read_bytes(),
        file_name="portfolio_6_tickets.csv",
        mime="text/csv",
    )

    st.subheader("📊 Résultat du backtest")

    backtest = results["backtest"]

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        st.metric(
            "Tirages évalués",
            backtest.get("draws", 0),
        )

    with metric2:
        st.metric(
            "Dépenses",
            f"{backtest.get('spent', 0):.2f} €",
        )

    with metric3:
        st.metric(
            "Gains indicatifs",
            f"{backtest.get('gains', 0):.2f} €",
        )

    with metric4:
        net_balance = backtest.get("net_balance", 0.0)
        st.metric(
            "Solde net",
            f"{net_balance:.2f} €",
            delta=f"{net_balance:.2f} €",
        )

    hit1, hit2 = st.columns(2)

    with hit1:
        st.info(f"🎯 Numéros trouvés : {backtest.get('number_hits', 0)}")

    with hit2:
        st.info(f"⭐ Étoiles trouvées : {backtest.get('star_hits', 0)}")

    st.caption(
        "Les gains du backtest sont indicatifs. Ils ne constituent pas "
        "une estimation fiable des gains futurs."
    )

else:
    st.info("Cliquez sur le bouton pour entraîner le modèle et générer les grilles.")
