"""
Application Web Streamlit - EuroMillions Pro IA
"""

from html import escape
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

        top_numbers_html = "".join(
            f"<span class='top-number-ball'>{int(number)}</span>"
            for number in results["top5_numbers"]
        )

        st.markdown(
            f"<div class='top-draw'>{top_numbers_html}</div>",
            unsafe_allow_html=True,
        )

    with col2:
        st.subheader("Top 2 étoiles apprises")

        top_stars_html = "".join(
            f"<span class='top-star-ball'>{int(star)}</span>"
            for star in results["top2_stars"]
        )

        st.markdown(
            f"<div class='top-draw'>{top_stars_html}</div>",
            unsafe_allow_html=True,
        )

    st.subheader("🎟️ Portefeuille généré")

    portfolio = pd.read_csv(results["portfolio_csv"])

    unnamed_columns = [
        column for column in portfolio.columns if column.lower().startswith("unnamed:")
    ]

    if unnamed_columns:
        portfolio = portfolio.drop(columns=unnamed_columns)

    column_labels = {
        "ticket": "Ticket",
        "n1": "Boule 1",
        "n2": "Boule 2",
        "n3": "Boule 3",
        "n4": "Boule 4",
        "n5": "Boule 5",
        "s1": "Étoile 1",
        "s2": "Étoile 2",
    }

    display_columns = [
        "ticket",
        "n1",
        "n2",
        "n3",
        "n4",
        "n5",
        "s1",
        "s2",
    ]

    headers = "".join(
        f"<th>{escape(column_labels[column])}</th>" for column in display_columns
    )

    rows_html = []

    for _, row in portfolio.iterrows():
        cells = [f"<td class='ticket-cell'>{int(row['ticket'])}</td>"]

        for column in ["n1", "n2", "n3", "n4", "n5"]:
            cells.append(
                f"<td><span class='number-ball'>{int(row[column])}</span></td>"
            )

        for column in ["s1", "s2"]:
            cells.append(f"<td><span class='star-ball'>{int(row[column])}</span></td>")

        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    portfolio_html = f"""
    <style>
        .portfolio-wrapper {{
            width: 100%;
            overflow-x: auto;
        }}

        .portfolio-table {{
            width: 100%;
            border-collapse: collapse;
            text-align: center;
            font-size: 16px;
        }}

        .portfolio-table th {{
            background-color: #f5f7fa;
            color: #555;
            padding: 12px 8px;
            border-bottom: 1px solid #dfe3e8;
            white-space: nowrap;
        }}

        .portfolio-table td {{
            height: 76px;
            padding: 8px;
            border-bottom: 1px solid #e5e7eb;
            text-align: center;
            vertical-align: middle;
        }}

        .ticket-cell {{
            font-weight: bold;
            color: #555;
        }}

        .number-ball,
        .star-ball {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 48px;
            height: 48px;
            color: white;
            font-weight: bold;
            font-size: 18px;
        }}

        .number-ball {{
            border-radius: 50%;
            background: #087caf;
        }}

        .star-ball {{
            background: #fbb64b;
            clip-path: polygon(
                50% 0%,
                61% 35%,
                98% 35%,
                68% 57%,
                79% 95%,
                50% 72%,
                21% 95%,
                32% 57%,
                2% 35%,
                39% 35%
            );
        }}
    </style>

    <div class="portfolio-wrapper">
        <table class="portfolio-table">
            <thead>
                <tr>{headers}</tr>
            </thead>
            <tbody>
                {"".join(rows_html)}
            </tbody>
        </table>
    </div>
    """

    st.markdown(
        portfolio_html,
        unsafe_allow_html=True,
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

    st.markdown(
        """
        <style>
            .top-draw {
                display: flex;
                gap: 12px;
                align-items: center;
                flex-wrap: wrap;
                margin-top: 12px;
                margin-bottom: 20px;
            }

            .top-number-ball,
            .top-star-ball {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 58px;
                height: 58px;
                color: white;
                font-size: 22px;
                font-weight: bold;
            }

            .top-number-ball {
                border-radius: 50%;
                background: #087caf;
            }

            .top-star-ball {
                background: #fbb64b;
                clip-path: polygon(
                    50% 0%,
                    61% 35%,
                    98% 35%,
                    68% 57%,
                    79% 95%,
                    50% 72%,
                    21% 95%,
                    32% 57%,
                    2% 35%,
                    39% 35%
                );
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

else:
    st.info("Cliquez sur le bouton pour entraîner le modèle et générer les grilles.")
