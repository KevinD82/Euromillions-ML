"""
Application Web Streamlit - EuroMillions Pro IA
"""

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from config import Config
from model.data_manager import DataManager
from model.euromillions_pro_pipeline import load_draws, run_pipeline

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

if not csv_file.exists():
    st.error(f"Fichier introuvable : {csv_file}")
    st.stop()

with st.spinner("Vérification des nouveaux tirages..."):
    try:
        synchronization = DataManager(str(csv_file)).synchronize_database()
        if synchronization.get("status") == "success":
            st.sidebar.success(
                f"{synchronization.get('added', 0)} nouveau(x) tirage(s) ajouté(s)."
            )
        elif synchronization.get("status") == "up_to_date":
            st.sidebar.caption("Historique synchronisé avec lesbonsnumeros_live.")
        else:
            st.sidebar.warning(
                f"Synchronisation indisponible : {synchronization.get('message', 'erreur inconnue')}"
            )
    except Exception as error:  # noqa: BLE001
        st.sidebar.warning(f"Synchronisation ignorée : {error}")

try:
    available_draws = load_draws(csv_file)
    last_history_date = available_draws["date"].max()
    st.sidebar.success(
        f"Dernier tirage chargé : {last_history_date.strftime('%d/%m/%Y')}"
    )

    earliest_target = pd.Timestamp(Config().model_start_date) + pd.Timedelta(
        days=Config().training_days
    )
    target_dates = list(
        available_draws.loc[
            available_draws["date"] >= earliest_target,
            "date",
        ].sort_values(ascending=False)
    )
    next_draw_date = last_history_date + pd.Timedelta(days=3)
    target_dates.insert(0, next_draw_date)

    def target_label(value):
        if value == next_draw_date:
            return f"Prochain tirage ({value.strftime('%d/%m/%Y')})"
        return value.strftime("Tirage du %d/%m/%Y")

    selected_target_date = st.sidebar.selectbox(
        "Date cible de la prédiction",
        options=target_dates,
        format_func=target_label,
    )
except Exception as error:  # noqa: BLE001
    st.error(f"Impossible de charger l’historique : {error}")
    st.stop()

run_button = st.sidebar.button(
    "🚀 Entraîner le modèle et générer les grilles",
    type="primary",
)
cache_file = (
    Path(output_dir) / f"prediction_{selected_target_date.strftime('%Y-%m-%d')}.json"
)
has_cached_result = cache_file.exists()

if run_button or has_cached_result:
    with st.spinner(
        "Entraînement annuel et backtest..."
        if run_button
        else "Chargement du résultat sauvegardé..."
    ):
        try:
            results = run_pipeline(
                str(csv_file),
                output_dir,
                Config(),
                target_date=selected_target_date.strftime("%Y-%m-%d"),
                force=run_button,
            )
        except Exception as error:  # noqa: BLE001
            st.error(f"Erreur pendant le pipeline : {error}")
            st.stop()

    if run_button:
        st.success("✅ Modèle entraîné et grilles générées.")
    else:
        st.info("Résultat sauvegardé chargé : aucun nouveau backtest exécuté.")

    st.caption(
        f"Prédiction pour le {pd.Timestamp(results['target_date']).strftime('%d/%m/%Y')} · "
        f"apprentissage du {pd.Timestamp(results['training_start']).strftime('%d/%m/%Y')} "
        f"au {pd.Timestamp(results['training_end']).strftime('%d/%m/%Y')}"
    )

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
