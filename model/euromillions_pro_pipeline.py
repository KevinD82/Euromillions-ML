"""
Pipeline EuroMillions avec apprentissage CatBoost.

Le modèle apprend uniquement à partir des tirages antérieurs à la cible.
Aucune méthode ne peut garantir les numéros gagnants.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from tqdm import tqdm

FEATURES = [
    "entity_id",
    "draw_index",
    "delay",
    "freq_5",
    "freq_10",
    "freq_25",
    "freq_50",
    "freq_all",
    "recent_weighted",
]

PAYOUTS = {
    (5, 2): 50_000_000.0,
    (5, 1): 250_000.0,
    (5, 0): 35_000.0,
    (4, 2): 1_500.0,
    (4, 1): 150.0,
    (4, 0): 60.0,
    (3, 2): 20.0,
    (2, 2): 15.0,
    (3, 1): 12.0,
    (3, 0): 10.0,
    (1, 2): 8.0,
    (2, 1): 6.0,
    (2, 0): 4.0,
}


def compute_official_euromillions_payout(
    matched_nums: int,
    matched_stars: int,
) -> float:
    return PAYOUTS.get((matched_nums, matched_stars), 0.0)


def load_draws(csv_path: str | Path) -> pd.DataFrame:
    """Charge et valide l'historique sans le modifier."""
    df = pd.read_csv(csv_path)

    if "date_de_tirage" in df.columns:
        date_column = "date_de_tirage"
    elif "date" in df.columns:
        date_column = "date"
    else:
        raise KeyError("Colonne de date introuvable.")

    number_columns = [
        "boule_1",
        "boule_2",
        "boule_3",
        "boule_4",
        "boule_5",
    ]
    star_columns = ["etoile_1", "etoile_2"]

    if not all(column in df.columns for column in number_columns):
        raise KeyError("Colonnes boule_1 à boule_5 introuvables.")

    if not all(column in df.columns for column in star_columns):
        raise KeyError("Colonnes etoile_1 et etoile_2 introuvables.")

    df = df.rename(
        columns={
            date_column: "date",
            **{column: f"n{index}" for index, column in enumerate(number_columns, 1)},
            "etoile_1": "s1",
            "etoile_2": "s2",
        }
    )

    df["date"] = pd.to_datetime(
        df["date"],
        format="%d/%m/%Y",
        errors="coerce",
    )

    if df["date"].isna().any():
        raise ValueError("Le CSV contient des dates invalides.")

    value_columns = [f"n{i}" for i in range(1, 6)] + ["s1", "s2"]

    for column in value_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df[value_columns].isna().any().any():
        raise ValueError("Le CSV contient des valeurs numériques invalides.")

    for line_number, (_, row) in enumerate(df.iterrows(), start=2):
        numbers = [int(row[f"n{i}"]) for i in range(1, 6)]
        stars = [int(row[f"s{i}"]) for i in range(1, 3)]

        if len(set(numbers)) != 5:
            raise ValueError(f"Numéros en doublon ligne {line_number}.")

        if not all(1 <= number <= 50 for number in numbers):
            raise ValueError(f"Numéro hors limites ligne {line_number}.")

        if len(set(stars)) != 2:
            raise ValueError(f"Étoiles en doublon ligne {line_number}.")

        if not all(1 <= star <= 12 for star in stars):
            raise ValueError(f"Étoile hors limites ligne {line_number}.")

    df = (
        df
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    df["draw_index"] = np.arange(len(df))
    return df


def _columns(entity_type: str) -> list[str]:
    if entity_type == "number":
        return [f"n{i}" for i in range(1, 6)]
    if entity_type == "star":
        return ["s1", "s2"]
    raise ValueError(f"Type inconnu : {entity_type}")


def _features_for_state(
    history: pd.DataFrame,
    pool_size: int,
    entity_type: str,
) -> pd.DataFrame:
    """Construit les caractéristiques après l'historique fourni."""
    columns = _columns(entity_type)
    draw_index = len(history)
    appearances = {entity: [] for entity in range(1, pool_size + 1)}

    for index, (_, row) in enumerate(history.iterrows()):
        for column in columns:
            entity = int(row[column])
            if entity in appearances:
                appearances[entity].append(index)

    rows = []

    for entity in range(1, pool_size + 1):
        positions = appearances[entity]

        row = {
            "entity_id": entity,
            "draw_index": draw_index,
            "delay": (draw_index - positions[-1] if positions else draw_index + 1),
            "freq_all": len(positions),
        }

        for window in (5, 10, 25, 50):
            row[f"freq_{window}"] = sum(
                draw_index - position <= window for position in positions
            )

        row["recent_weighted"] = sum(
            np.exp(-(draw_index - position) / 20.0) for position in positions
        )

        rows.append(row)

    return pd.DataFrame(rows, columns=FEATURES)


def build_long_table(
    df_draws: pd.DataFrame,
    pool_size: int,
    entity_type: str,
) -> pd.DataFrame:
    """
    Construit les exemples d'apprentissage.

    Les caractéristiques du tirage i utilisent uniquement les tirages
    précédant i.
    """
    columns = _columns(entity_type)
    rows = []

    for target_index in range(1, len(df_draws)):
        history = df_draws.iloc[:target_index]
        features = _features_for_state(
            history,
            pool_size,
            entity_type,
        )

        target = df_draws.iloc[target_index]
        winners = {int(target[column]) for column in columns}

        features["label"] = features["entity_id"].isin(winners).astype(int)
        features["target_index"] = target_index
        rows.append(features)

    if not rows:
        return pd.DataFrame(columns=FEATURES + ["label", "target_index"])

    return pd.concat(rows, ignore_index=True)


def build_next_draw_features(
    df_draws: pd.DataFrame,
    pool_size: int,
    entity_type: str,
) -> pd.DataFrame:
    return _features_for_state(
        df_draws,
        pool_size,
        entity_type,
    )


def train_entity_model(
    training_table: pd.DataFrame,
    cfg,
) -> CatBoostClassifier:
    if training_table.empty:
        raise ValueError("Table d'apprentissage vide.")

    # Limite de sécurité pour accélérer le backtest.
    max_rows = 30_000
    if len(training_table) > max_rows:
        training_table = training_table.tail(max_rows)

    model = CatBoostClassifier(**cfg.catboost_params)

    model.fit(
        training_table[FEATURES],
        training_table["label"],
    )

    return model


def predict_entity_probabilities(
    model: CatBoostClassifier,
    features: pd.DataFrame,
    pool_size: int,
) -> dict[int, float]:
    probabilities = model.predict_proba(features[FEATURES])[:, 1]

    # Mélange prudent : 80 % prédiction ML, 20 % uniforme.
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.nan_to_num(
        probabilities,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    probabilities += 0.20 / pool_size
    total = probabilities.sum()

    if total <= 0:
        probabilities = np.full(pool_size, 1.0 / pool_size)
    else:
        probabilities /= total

    return {
        int(entity): float(probability)
        for entity, probability in zip(
            features["entity_id"],
            probabilities,
        )
    }


def train_and_predict(
    history: pd.DataFrame,
    cfg,
) -> tuple[dict[int, float], dict[int, float]]:
    print(
        f"Construction des données numéros ({len(history)} tirages)...",
        flush=True,
    )

    number_table = build_long_table(
        history,
        cfg.pool_numbers,
        "number",
    )

    print(
        f"Apprentissage CatBoost numéros ({len(number_table)} lignes)...",
        flush=True,
    )

    number_model = train_entity_model(
        number_table,
        cfg,
    )

    print(
        "Construction des données étoiles...",
        flush=True,
    )

    star_table = build_long_table(
        history,
        cfg.pool_stars,
        "star",
    )

    print(
        f"Apprentissage CatBoost étoiles ({len(star_table)} lignes)...",
        flush=True,
    )

    star_model = train_entity_model(
        star_table,
        cfg,
    )

    number_features = build_next_draw_features(
        history,
        cfg.pool_numbers,
        "number",
    )
    star_features = build_next_draw_features(
        history,
        cfg.pool_stars,
        "star",
    )

    return (
        predict_entity_probabilities(
            number_model,
            number_features,
            cfg.pool_numbers,
        ),
        predict_entity_probabilities(
            star_model,
            star_features,
            cfg.pool_stars,
        ),
    )


def generate_ticket(
    number_probabilities: dict[int, float],
    star_probabilities: dict[int, float],
    rng: np.random.Generator,
) -> dict:
    numbers = np.array(list(number_probabilities))
    stars = np.array(list(star_probabilities))

    number_weights = np.array(
        list(number_probabilities.values()),
        dtype=float,
    )
    star_weights = np.array(
        list(star_probabilities.values()),
        dtype=float,
    )

    number_weights /= number_weights.sum()
    star_weights /= star_weights.sum()

    selected_numbers = sorted(
        rng.choice(
            numbers,
            5,
            replace=False,
            p=number_weights,
        ).tolist()
    )

    selected_stars = sorted(
        rng.choice(
            stars,
            2,
            replace=False,
            p=star_weights,
        ).tolist()
    )

    return {
        "nums": selected_numbers,
        "stars": selected_stars,
        "score": sum(number_probabilities[number] for number in selected_numbers),
    }


def build_combined_grid(
    grids: list[dict],
    number_probabilities: dict[int, float],
    star_probabilities: dict[int, float],
) -> dict:
    number_counts = {number: 0 for number in number_probabilities}
    star_counts = {star: 0 for star in star_probabilities}

    for grid in grids:
        for number in grid["nums"]:
            number_counts[number] += 1
        for star in grid["stars"]:
            star_counts[star] += 1

    numbers = sorted(
        number_probabilities,
        key=lambda value: (
            number_counts[value],
            number_probabilities[value],
        ),
        reverse=True,
    )[:5]

    stars = sorted(
        star_probabilities,
        key=lambda value: (
            star_counts[value],
            star_probabilities[value],
        ),
        reverse=True,
    )[:2]

    return {
        "nums": sorted(numbers),
        "stars": sorted(stars),
        "score": 0.0,
    }


def backtest(
    df_draws: pd.DataFrame,
    cfg,
    out_dir: str,
) -> dict:
    """Évalue les 20 derniers tirages de l'année précédente."""
    minimum = max(cfg.min_history_draws, 30)

    if len(df_draws) <= minimum:
        return {
            "draws": 0,
            "spent": 0.0,
            "gains": 0.0,
            "net_balance": 0.0,
            "number_hits": 0,
            "star_hits": 0,
        }

    last_date = df_draws["date"].max()
    first_date = last_date - pd.Timedelta(days=cfg.backtest_days)

    candidate_indices = [
        index
        for index, date in enumerate(df_draws["date"])
        if first_date < date <= last_date and index >= minimum
    ]

    if not candidate_indices:
        return {
            "draws": 0,
            "spent": 0.0,
            "gains": 0.0,
            "net_balance": 0.0,
            "number_hits": 0,
            "star_hits": 0,
        }

    # Les 20 tirages les plus récents de la période sélectionnée.
    target_indices = candidate_indices[-cfg.n_splits_backtest :]

    total_spent = 0.0
    total_gains = 0.0
    number_hits = 0
    star_hits = 0

    print(
        f"📊 Backtest sur {len(target_indices)} tirages "
        f"entre {first_date.date()} et {last_date.date()}",
        flush=True,
    )

    for position, target_index in enumerate(
        tqdm(
            target_indices,
            desc="Backtest année précédente",
            total=len(target_indices),
        ),
        start=1,
    ):
        history_start = max(
            0,
            target_index - cfg.backtest_training_draws,
        )

        history = df_draws.iloc[history_start:target_index].copy()

        actual = df_draws.iloc[target_index]

        print(
            f"Entraînement {position}/{len(target_indices)} "
            f"- historique utilisé : {len(history)} tirages",
            flush=True,
        )

        numbers, stars = train_and_predict(history, cfg)
        rng = np.random.default_rng(1000 + target_index)

        actual_numbers = {int(actual[f"n{i}"]) for i in range(1, 6)}
        actual_stars = {int(actual[f"s{i}"]) for i in range(1, 3)}

        for _ in range(cfg.backtest_tickets_per_draw):
            ticket = generate_ticket(numbers, stars, rng)

            matched_numbers = len(actual_numbers & set(ticket["nums"]))
            matched_stars = len(actual_stars & set(ticket["stars"]))

            number_hits += matched_numbers
            star_hits += matched_stars

            total_gains += compute_official_euromillions_payout(
                matched_numbers,
                matched_stars,
            )

        total_spent += 2.50 * cfg.backtest_tickets_per_draw

    return {
        "draws": len(target_indices),
        "period_start": str(first_date.date()),
        "period_end": str(last_date.date()),
        "spent": total_spent,
        "gains": total_gains,
        "net_balance": total_gains - total_spent,
        "number_hits": number_hits,
        "star_hits": star_hits,
    }


def run_pipeline(
    csv_path: str,
    out_dir: str,
    cfg,
) -> dict:
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df_draws = load_draws(csv_path)

    if len(df_draws) < cfg.min_history_draws:
        raise ValueError(f"Il faut au moins {cfg.min_history_draws} tirages.")

    backtest_result = backtest(
        df_draws,
        cfg,
        out_dir,
    )

    number_probabilities, star_probabilities = train_and_predict(
        df_draws,
        cfg,
    )

    print(
        "✅ Entraînement final terminé. Génération des grilles...",
        flush=True,
    )

    top5_numbers = sorted(
        sorted(
            number_probabilities,
            key=number_probabilities.get,
            reverse=True,
        )[:5]
    )

    top2_stars = sorted(
        sorted(
            star_probabilities,
            key=star_probabilities.get,
            reverse=True,
        )[:2]
    )

    rng = np.random.default_rng(2026)
    grids = []

    for _ in range(cfg.generated_tickets):
        grids.append(
            generate_ticket(
                number_probabilities,
                star_probabilities,
                rng,
            )
        )

    grids.append(
        build_combined_grid(
            grids,
            number_probabilities,
            star_probabilities,
        )
    )

    portfolio = []

    for ticket, grid in enumerate(grids, start=1):
        portfolio.append({
            "ticket": ticket,
            "n1": grid["nums"][0],
            "n2": grid["nums"][1],
            "n3": grid["nums"][2],
            "n4": grid["nums"][3],
            "n5": grid["nums"][4],
            "s1": grid["stars"][0],
            "s2": grid["stars"][1],
        })

    portfolio_path = output_path / "portfolio_6_tickets.csv"
    pd.DataFrame(portfolio).to_csv(
        portfolio_path,
        index=False,
    )

    return {
        "top5_numbers": top5_numbers,
        "top2_stars": top2_stars,
        "portfolio_csv": str(portfolio_path),
        "backtest": backtest_result,
    }

    final_history = df_draws.tail(cfg.final_training_draws).reset_index(drop=True)

    print(
        f"✅ Backtest terminé. Entraînement final sur {len(final_history)} tirages...",
        flush=True,
    )

    number_probabilities, star_probabilities = train_and_predict(
        final_history,
        cfg,
    )
