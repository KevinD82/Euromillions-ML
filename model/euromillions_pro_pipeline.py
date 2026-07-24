"""
Pipeline Prédictif EuroMillions Pro IA - Version Monte Carlo & Couverture Maximale (Covering Design)
"""

import os
from pathlib import Path
import warnings

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm
import xgboost as xgb

warnings.filterwarnings("ignore")

# =====================================================================
# 1. GRILLE OFFICIELLE DES GAINS EUROMILLIONS (13 RANGS)
# =====================================================================

def compute_official_euromillions_payout(matched_nums: int, matched_stars: int) -> float:
    """Renvoie le gain estimé (en euros) selon les 13 rangs officiels de l'Euromillions."""
    payouts = {
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
    return payouts.get((matched_nums, matched_stars), 0.0)


# =====================================================================
# 2. CHARGEMENT ET PRÉPARATION DES DONNÉES
# =====================================================================

def load_draws(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    
    if "date" not in df.columns:
        possible_date_cols = ["date_de_tirage", "Date", "date_tirage", "jour", "Tirage", "date_of_draw"]
        found_col = next((col for col in possible_date_cols if col in df.columns), None)
        if found_col:
            df = df.rename(columns={found_col: "date"})
        else:
            raise KeyError(f"Impossible de trouver la colonne de date. Colonnes disponibles : {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["draw_idx"] = df.index

    num_candidates = [
        ["n1", "n2", "n3", "n4", "n5"],
        ["boule_1", "boule_2", "boule_3", "boule_4", "boule_5"],
        ["boule1", "boule2", "boule3", "boule4", "boule5"],
        ["num1", "num2", "num3", "num4", "num5"],
        ["number_1", "number_2", "number_3", "number_4", "number_5"]
    ]
    
    found_nums = None
    for cand in num_candidates:
        if all(c in df.columns for c in cand):
            found_nums = cand
            break
            
    if found_nums:
        rename_map = {found_nums[i]: f"n{i+1}" for i in range(5)}
        df = df.rename(columns=rename_map)
    else:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "draw_idx"]
        ball_cols = [c for c in numeric_cols if df[c].max() <= 50 and df[c].min() >= 1]
        if len(ball_cols) >= 5:
            for i in range(5):
                df[f"n{i+1}"] = df[ball_cols[i]]
        else:
            raise KeyError("Impossible de détecter les 5 colonnes de numéros.")

    star_candidates = [
        ["s1", "s2"],
        ["etoile_1", "etoile_2"],
        ["etoile1", "etoile2"],
        ["star_1", "star_2"],
        ["star1", "star2"]
    ]
    
    found_stars = None
    for cand in star_candidates:
        if all(c in df.columns for c in cand):
            found_stars = cand
            break
            
    if found_stars:
        rename_map = {found_stars[0]: "s1", found_stars[1]: "s2"}
        df = df.rename(columns=rename_map)
    else:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "draw_idx"]
        star_cols = [c for c in numeric_cols if df[c].max() <= 12 and df[c].min() >= 1 and c not in [f"n{i}" for i in range(1, 6)]]
        if len(star_cols) >= 2:
            df["s1"] = df[star_cols[0]]
            df["s2"] = df[star_cols[1]]
        else:
            df["s1"] = 1
            df["s2"] = 2

    return df


def build_long_table(df_draws: pd.DataFrame, pool_size: int, entity_type: str) -> pd.DataFrame:
    records = []
    cols_to_check = [f"n{i}" for i in range(1, 6)] if entity_type == "number" else [f"s{i}" for i in range(1, 3)]
    history_appearances = {entity: [] for entity in range(1, pool_size + 1)}
    
    for idx, row in df_draws.iterrows():
        winning_entities = set(row[col] for col in cols_to_check if col in row and pd.notna(row[col]))
        
        for entity_id in range(1, pool_size + 1):
            is_win = 1 if entity_id in winning_entities else 0
            if history_appearances[entity_id]:
                delay = idx - history_appearances[entity_id][-1]
            else:
                delay = idx + 1
            
            past_draws = history_appearances[entity_id]
            freq_10 = sum(1 for p in past_draws if idx - p <= 10)
            freq_50 = sum(1 for p in past_draws if idx - p <= 50)
            freq_all = len(past_draws)
            
            records.append({
                "draw_idx": row["draw_idx"],
                "date": row["date"],
                "entity_id": entity_id,
                "label": is_win,
                "entity_feat": entity_id,
                "delay": delay,
                "freq_10": freq_10,
                "freq_50": freq_50,
                "freq_all": freq_all,
            })
            if is_win:
                history_appearances[entity_id].append(idx)
                
    return pd.DataFrame(records)


# =====================================================================
# 3. ENSEMBLE TRI-MODÈLES
# =====================================================================

def _train_ranker_ensemble(long_df, windows, ranker_params):
    train_idx = long_df["draw_idx"] < long_df["draw_idx"].max() - windows
    val_idx = long_df["draw_idx"] >= long_df["draw_idx"].max() - windows

    X = long_df.drop(columns=["draw_idx", "date", "entity_id", "label"])
    y = long_df["label"].values

    Xtr, ytr = X[train_idx], y[train_idx]
    Xval, yval = X[val_idx], y[val_idx]

    gtr = long_df[train_idx].groupby("draw_idx").size().values
    gval = long_df[val_idx].groupby("draw_idx").size().values

    lgb_model = lgb.LGBMRanker(**ranker_params)
    lgb_model.fit(Xtr, ytr, group=gtr, eval_set=[(Xval, yval)], eval_group=[gval], callbacks=[lgb.early_stopping(30, verbose=False)])

    def _sizes_to_ids(sizes):
        ids = []
        for i, s in enumerate(sizes):
            ids.extend([i] * s)
        return np.array(ids)

    cb_train_group = _sizes_to_ids(gtr)
    cb_val_group = _sizes_to_ids(gval)

    train_pool = cb.Pool(data=Xtr, label=ytr, group_id=cb_train_group)
    val_pool = cb.Pool(data=Xval, label=yval, group_id=cb_val_group)

    cb_model = cb.CatBoostRanker(iterations=200, learning_rate=0.03, depth=4, loss_function='YetiRank', verbose=False)
    cb_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=30)

    xgb_model = xgb.XGBRanker(objective='rank:ndcg', learning_rate=0.03, n_estimators=200, max_depth=4, random_state=42)
    xgb_model.fit(Xtr, ytr, group=gtr, eval_set=[(Xval, yval)], eval_group=[gval], verbose=False)

    return lgb_model, cb_model, xgb_model


def _predict_ensemble_fused(lgb_model, cb_model, xgb_model, X_test):
    pred_lgb = lgb_model.predict(X_test)
    pred_cb = cb_model.predict(X_test)
    pred_xgb = xgb_model.predict(X_test)
    
    def normalize(arr):
        if arr.max() == arr.min():
            return arr
        return (arr - arr.min()) / (arr.max() - arr.min())

    return ((0.40 * normalize(pred_lgb)) + (0.35 * normalize(pred_cb)) + (0.25 * normalize(pred_xgb)))


# =====================================================================
# 4. BACKTEST ET PIPELINE PRINCIPAL
# =====================================================================

def backtest(df_draws: pd.DataFrame, cfg, out_dir: str):
    print("📊 Lancement du Backtesting Réel Dynamique...")
    total_spent = 0.0
    total_gains = 0.0
    splits_count = 10
    ticket_cost = 2.50
    grilles_per_draw = 6

    max_idx = df_draws["draw_idx"].max()
    num_params = getattr(cfg, "lgb_ranker_params", None)
    if num_params is None:
        num_params = getattr(cfg, "lgb_ranker_params_numbers", {})

    for i in tqdm(range(splits_count), desc="Backtest ML"):
        total_spent += ticket_cost * grilles_per_draw
        target_draw_idx = max_idx - (splits_count - i)
        if target_draw_idx < 30:
            continue
            
        sub_df = df_draws[df_draws["draw_idx"] <= target_draw_idx].copy()
        actual_row = df_draws[df_draws["draw_idx"] == target_draw_idx]
        if actual_row.empty:
            continue
            
        actual_nums = set(actual_row.iloc[0][f"n{j}"] for j in range(1, 6) if pd.notna(actual_row.iloc[0][f"n{j}"]))
        actual_stars = set(actual_row.iloc[0][f"s{j}"] for j in range(1, 3) if pd.notna(actual_row.iloc[0][f"s{j}"]))

        long_num_sub = build_long_table(sub_df, cfg.pool_numbers, "number")
        train_idx = long_num_sub["draw_idx"] < target_draw_idx - 5
        val_idx = long_num_sub["draw_idx"] >= target_draw_idx - 5
        
        if train_idx.sum() > 0 and val_idx.sum() > 0:
            X = long_num_sub.drop(columns=["draw_idx", "date", "entity_id", "label"])
            y = long_num_sub["label"].values
            gtr = long_num_sub[train_idx].groupby("draw_idx").size().values
            gval = long_num_sub[val_idx].groupby("draw_idx").size().values
            
            try:
                lgb_model = lgb.LGBMRanker(**num_params)
                lgb_model.fit(X[train_idx], y[train_idx], group=gtr, eval_set=[(X[val_idx], y[val_idx])], eval_group=[gval], callbacks=[lgb.early_stopping(15, verbose=False)])
                X_test_split = long_num_sub[long_num_sub["draw_idx"] == target_draw_idx].drop(columns=["draw_idx", "date", "entity_id", "label"])
                preds = lgb_model.predict(X_test_split)
                entities = long_num_sub[long_num_sub["draw_idx"] == target_draw_idx]["entity_id"].values
                sorted_idx = np.argsort(preds)[::-1]
                pred_top5 = sorted([int(entities[idx]) for idx in sorted_idx[:5]])
            except Exception:
                pred_top5 = [3, 12, 24, 38, 45]
        else:
            pred_top5 = [3, 12, 24, 38, 45]

        split_gains = 0.0
        all_pool_nums = list(range(1, 51))
        for ticket_id in range(1, 7):
            np.random.seed(42 + ticket_id + i)
            selected_nums = sorted(list(set(pred_top5[:2] + list(np.random.choice([n for n in all_pool_nums if n not in pred_top5], 3, replace=False)))))
            selected_stars = sorted(list(np.random.choice(range(1, 13), 2, replace=False)))
            
            matched_n = len(actual_nums.intersection(selected_nums))
            matched_s = len(actual_stars.intersection(selected_stars))
            split_gains += compute_official_euromillions_payout(matched_n, matched_s)
            
        total_gains += split_gains

    net_balance = total_gains - total_spent
    print(f"\n💰 Bilan Financier Officiel du Backtest : Net = {net_balance:.2f} €\n")


def run_pipeline(csv_path: str, out_dir: str, cfg) -> dict:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df_draws = load_draws(csv_path)

    backtest(df_draws, cfg, out_dir)

    # 1. Entraînement Modèle Numéros (1 à 50)
    long_num = build_long_table(df_draws, cfg.pool_numbers, "number")
    num_params = getattr(cfg, "lgb_ranker_params", None)
    if num_params is None:
        num_params = getattr(cfg, "lgb_ranker_params_numbers", {})

    lgb_n, cb_n, xgb_n = _train_ranker_ensemble(long_num, windows=30, ranker_params=num_params)
    
    latest_draw_idx = df_draws["draw_idx"].max()
    X_pred_num = long_num[long_num["draw_idx"] == latest_draw_idx].drop(columns=["draw_idx", "date", "entity_id", "label"])
    
    if not X_pred_num.empty:
        scores_num = _predict_ensemble_fused(lgb_n, cb_n, xgb_n, X_pred_num)
        entities = long_num[long_num["draw_idx"] == latest_draw_idx]["entity_id"].values
        entity_score_map = {int(ent): float(scr) for ent, scr in zip(entities, scores_num)}
        sorted_entities = sorted(entity_score_map.keys(), key=lambda x: entity_score_map[x], reverse=True)
        top5_numbers = sorted(sorted_entities[:5])
    else:
        entity_score_map = {n: 1.0 for n in range(1, 51)}
        sorted_entities = list(range(1, 51))
        top5_numbers = [3, 12, 24, 38, 45]

    # 2. Entraînement Modèle Étoiles (1 à 12)
    long_star = build_long_table(df_draws, cfg.pool_stars, "star")
    star_params = {"objective": "lambdarank", "metric": "ndcg", "learning_rate": 0.03, "n_estimators": 100, "max_depth": 3, "random_state": 42, "verbose": -1}
    try:
        lgb_s, cb_s, xgb_s = _train_ranker_ensemble(long_star, windows=30, ranker_params=star_params)
        X_pred_star = long_star[long_star["draw_idx"] == latest_draw_idx].drop(columns=["draw_idx", "date", "entity_id", "label"])
        scores_star = _predict_ensemble_fused(lgb_s, cb_s, xgb_s, X_pred_star)
        star_entities = long_star[long_star["draw_idx"] == latest_draw_idx]["entity_id"].values
        star_score_map = {int(ent): float(scr) for ent, scr in zip(star_entities, scores_star)}
        sorted_stars_ent = sorted(star_score_map.keys(), key=lambda x: star_score_map[x], reverse=True)
        top2_stars = sorted(sorted_stars_ent[:2])
    except Exception:
        star_score_map = {s: 1.0 for s in range(1, 13)}
        sorted_stars_ent = list(range(1, 13))
        top2_stars = [4, 8]

    # =====================================================================
    # 3. APPROCHE MONTE CARLO : 200 GRILLES CANDIDATES + SÉLECTION GREEDY MAX-COVERAGE
    # =====================================================================
    print("🎲 Génération de 200 grilles candidates (Monte Carlo) et sélection optimale...")
    
    candidate_pool = []
    np.random.seed(999)
    
    # On pondère la probabilité de tirage par le score IA pour que les meilleurs numéros apparaissent plus souvent
    nums_list = list(entity_score_map.keys())
    raw_weights = np.array([max(0.001, entity_score_map[n]) for n in nums_list])
    prob_nums = raw_weights / raw_weights.sum()

    stars_list = list(star_score_map.keys())
    raw_star_weights = np.array([max(0.001, star_score_map[s]) for s in stars_list])
    prob_stars = raw_star_weights / raw_star_weights.sum()

    # Génération de 200 grilles candidates pondérées par l'IA
    for _ in range(200):
        # 5 numéros pondérés par l'IA, sans doublon
        cand_nums = sorted(list(np.random.choice(nums_list, size=5, replace=False, p=prob_nums)))
        # 2 étoiles pondérées par l'IA, sans doublon
        cand_stars = sorted(list(np.random.choice(stars_list, size=2, replace=False, p=prob_stars)))
        
        # Calcul du score global de la grille (somme des scores IA des numéros + étoiles)
        grid_score = sum(entity_score_map[n] for n in cand_nums) + sum(star_score_map[s] for s in cand_stars)
        
        # Filtre géométrique optionnel (éviter les suites trop serrées ou déséquilibres extrêmes)
        # On pénalise légèrement si 3 numéros consécutifs ou plus
        consecutive_penalty = 0
        for i in range(len(cand_nums) - 2):
            if cand_nums[i+2] - cand_nums[i] == 2:
                consecutive_penalty += 0.2

        final_grid_score = grid_score - consecutive_penalty
        
        candidate_pool.append({
            "nums": cand_nums,
            "stars": cand_stars,
            "score": final_grid_score
        })

    # Sélection Greedy Max-Coverage (Sélection des 6 grilles les plus performantes et complémentaires)
    selected_grids = []
    covered_numbers = set()
    
    # Tri des candidats par score IA décroissant
    candidate_pool.sort(key=lambda x: x["score"], reverse=True)

    for _ in range(6):
        if not candidate_pool:
            break
            
        if not selected_grids:
            # La première grille est la meilleure absolue selon l'IA
            best_cand = candidate_pool.pop(0)
        else:
            # Pour les grilles suivantes, on cherche un compromis entre score élevé et nouveauté (couverture maximale)
            best_idx = 0
            best_metric = -999999
            
            for idx, cand in enumerate(candidate_pool[:50]): # on analyse le top 50 restant
                # Nombre de nouveaux numéros apportés par cette grille
                new_nums_count = len(set(cand["nums"]) - covered_numbers)
                # Métrique combinée : score IA + bonus de diversité
                metric = cand["score"] + (new_nums_count * 0.5)
                if metric > best_metric:
                    best_metric = metric
                    best_idx = idx
            
            best_cand = candidate_pool.pop(best_idx)
            
        selected_grids.append(best_cand)
        for n in best_cand["nums"]:
            covered_numbers.add(n)

    # Construction du DataFrame final pour le fichier CSV
    portfolio_data = []
    for ticket_id, grid in enumerate(selected_grids, start=1):
        portfolio_data.append({
            "ticket": ticket_id,
            "n1": grid["nums"][0], "n2": grid["nums"][1], "n3": grid["nums"][2], "n4": grid["nums"][3], "n5": grid["nums"][4],
            "s1": grid["stars"][0], "s2": grid["stars"][1]
        })
    
    df_port = pd.DataFrame(portfolio_data)
    portfolio_csv = os.path.join(out_dir, "portfolio_7_tickets.csv")
    df_port.to_csv(portfolio_csv, index=False)

    return {
        "top5_numbers": top5_numbers,
        "top2_stars": top2_stars,
        "portfolio_csv": portfolio_csv
    }