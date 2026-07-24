#!/usr/bin/env python3
"""
EuroMillions Pro — Pipeline ML Optimisé (LightGBM + Random Forest + Gaps Avancés + Backtest Financier)
Architecture MVC Propre & Haute Performance
"""

# ================================
# 📦 IMPORTS OPTIMISÉS
# ================================
import gc
import sys
from math import log2
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

# Ajoute le dossier racine au path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from config import Config
from utils.cache_manager import CacheManager

cache = CacheManager()


# ================================
# 📥 CHARGEMENT CSV + MAPPING
# ================================
COLUMN_MAPPING_FR = {
    "date_de_tirage": "date",
    "boule_1": "n1",
    "boule_2": "n2",
    "boule_3": "n3",
    "boule_4": "n4",
    "boule_5": "n5",
    "etoile_1": "s1",
    "etoile_2": "s2",
}


def _read_csv_auto(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, ValueError):
        return pd.read_csv(path, sep=";")


def load_draws(csv_path: str) -> pd.DataFrame:
    df = _read_csv_auto(csv_path)

    rename = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc in COLUMN_MAPPING_FR:
            rename[c] = COLUMN_MAPPING_FR[lc]

    df = df.rename(columns=rename)
    df.columns = [c.strip().lower() for c in df.columns]

    needed = ["date", "n1", "n2", "n3", "n4", "n5", "s1", "s2"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df = df[df["date"] >= (pd.Timestamp.now() - pd.DateOffset(years=5))]

    num_cols = ["n1", "n2", "n3", "n4", "n5", "s1", "s2"]
    df = df.dropna(subset=num_cols)
    df[num_cols] = df[num_cols].astype(int)
    df = df.sort_values("date").reset_index(drop=True)

    return df


# ================================
# 🧠 FEATURES AVANCÉES (Gaps & Cycles)
# ================================
def _exp_decay_factor(half_life: int) -> float:
    return 0.5 ** (1.0 / max(1, half_life))


def _presence_matrix(df: pd.DataFrame, pool: int, cols: list[str]) -> np.ndarray:
    n = len(df)
    P = np.zeros((n, pool), dtype=np.int8)
    for i, row in df.iterrows():
        for c in cols:
            v = row[c]
            if 1 <= v <= pool:
                P[i, v - 1] = 1
    return P


def _cooc_decay_features(
    df: pd.DataFrame, pool: int, cols: list[str], half_life: int, topk: int
):
    n = len(df)
    rho = _exp_decay_factor(half_life)

    C = np.zeros((pool, pool), dtype=np.float32)
    cooc_sum = np.zeros((n, pool), dtype=np.float32)
    cooc_topk_mean = np.zeros((n, pool), dtype=np.float32)

    for i, row in df.iterrows():
        cooc_sum[i] = C.sum(axis=1)

        if topk > 0:
            sorted_rows = np.sort(C, axis=1)
            cooc_topk_mean[i] = sorted_rows[:, -topk:].mean(axis=1)

        picks = [row[c] for c in cols]
        idx = [p - 1 for p in picks if 1 <= p <= pool]

        A = np.zeros((pool, pool), dtype=np.float32)
        for a in idx:
            for b in idx:
                if a != b:
                    A[a, b] = 1.0

        C = rho * C + A

    return cooc_sum, cooc_topk_mean


# ================================
# 🧠 HELPERS GÉNÉRAUX
# ================================
def _maybe_gpu(params: dict, cfg: Config) -> dict:
    p = params.copy()
    if hasattr(cfg, "gpu_try") and cfg.gpu_try:
        p["device"] = "gpu"
    return p


def _sample_weights_by_recency(draw_idx: np.ndarray, half_life: int) -> np.ndarray:
    last = np.max(draw_idx)
    age = last - draw_idx
    return 0.5 ** (age / float(max(1, half_life)))


def _minmax01(x: np.ndarray) -> np.ndarray:
    a, b = float(np.min(x)), float(np.max(x))
    if b - a < 1e-12:
        return np.ones_like(x) * 0.5
    return (x - a) / (b - a)


# ================================
# 🧱 CONSTRUCTION TABLE LONGUE + GAPS AVANCÉS
# ================================
def build_long_table(
    df_draws: pd.DataFrame, pool: int, kind: str, cfg: Config
) -> pd.DataFrame:
    assert kind in ("number", "star")

    cols = ["n1", "n2", "n3", "n4", "n5"] if kind == "number" else ["s1", "s2"]
    df = df_draws.copy().reset_index(drop=True)

    if not cfg.allow_older_regimes:
        mask_ok = df[cols].le(pool).all(axis=1)
        df = df.loc[mask_ok].reset_index(drop=True)

    n = len(df)
    if n < cfg.min_history_draws:
        raise ValueError(f"Pas assez de tirages ({n}) ; min={cfg.min_history_draws}.")

    P = _presence_matrix(df, pool, cols)
    Pm1 = np.vstack([np.zeros((1, pool), dtype=np.int8), P[:-1]])

    Pdf = pd.DataFrame(Pm1)
    cnt_w = {}
    rate_w = {}

    windows = (10, 25, 50, 100, 200)
    for w in windows:
        c = Pdf.rolling(window=w, min_periods=1).sum().to_numpy(dtype=np.float32)
        cnt_w[w] = c
        denom = np.minimum(w, np.arange(n)[:, None])
        denom[denom == 0] = 1
        rate_w[w] = c / denom

    ewma_s = {}
    for s in (10, 25, 50, 100):
        ewma_s[s] = Pdf.ewm(span=s, adjust=False).mean().to_numpy(dtype=np.float32)

    # ➡️ Utilisation de int32 pour éviter l'overflow des gaps
    gap = np.full((n, pool), 9999, dtype=np.int32)
    max_gap = np.zeros((n, pool), dtype=np.int32)
    gap_mean = np.zeros((n, pool), dtype=np.float32)
    gap_std = np.zeros((n, pool), dtype=np.float32)

    last = np.full(pool, -1, dtype=np.int32)
    all_historical_gaps = [[] for _ in range(pool)]

    for i in range(n):
        current_gaps = i - last
        gap[i] = current_gaps

        for j in range(pool):
            all_historical_gaps[j].append(current_gaps[j])
            max_gap[i, j] = int(np.max(all_historical_gaps[j]))
            gap_mean[i, j] = float(np.mean(all_historical_gaps[j]))
            gap_std[i, j] = (
                float(np.std(all_historical_gaps[j]))
                if len(all_historical_gaps[j]) > 1
                else 0.0
            )

        idx1 = np.where(Pm1[i] == 1)[0]
        if idx1.size:
            last[idx1] = i

    streak5 = Pdf.rolling(window=5, min_periods=1).sum().to_numpy(dtype=np.float32)
    age = np.arange(n).astype(np.float32)[:, None] * np.ones(
        (1, pool), dtype=np.float32
    )

    def exp_decay_series(Pshift: np.ndarray, half_life: int) -> np.ndarray:
        rho = _exp_decay_factor(half_life)
        E = np.zeros_like(Pshift, dtype=np.float32)
        for i in range(1, n):
            E[i] = rho * E[i - 1] + Pshift[i - 1]
        return E

    if kind == "number":
        Edec = exp_decay_series(Pm1, cfg.half_life_numbers)
        cooc_sum, cooc_topk = _cooc_decay_features(
            df, pool, cols, cfg.cooc_half_life_numbers, cfg.cooc_topk
        )
    else:
        Edec = exp_decay_series(Pm1, cfg.half_life_stars)
        cooc_sum, cooc_topk = _cooc_decay_features(
            df, pool, cols, cfg.cooc_half_life_stars, cfg.cooc_topk
        )

    rows = []
    for i in range(n):
        date = df.loc[i, "date"]
        for j in range(pool):
            rows.append({
                "draw_idx": i,
                "date": date,
                "entity_id": j + 1,
                "label": int(P[i, j] == 1),
                "cnt_w10": float(cnt_w[10][i, j]),
                "cnt_w25": float(cnt_w[25][i, j]),
                "cnt_w50": float(cnt_w[50][i, j]),
                "cnt_w100": float(cnt_w[100][i, j]),
                "cnt_w200": float(cnt_w[200][i, j]),
                "rate_w10": float(rate_w[10][i, j]),
                "rate_w25": float(rate_w[25][i, j]),
                "rate_w50": float(rate_w[50][i, j]),
                "rate_w100": float(rate_w[100][i, j]),
                "rate_w200": float(rate_w[200][i, j]),
                "ewma_s10": float(ewma_s[10][i, j]),
                "ewma_s25": float(ewma_s[25][i, j]),
                "ewma_s50": float(ewma_s[50][i, j]),
                "ewma_s100": float(ewma_s[100][i, j]),
                "gap_draws": int(gap[i, j]),
                "max_gap": int(max_gap[i, j]),
                "gap_mean": float(gap_mean[i, j]),
                "gap_std": float(gap_std[i, j]),
                "streak_5_sum": float(streak5[i, j]),
                "age_draws": float(age[i, j]),
                "expdecay": float(Edec[i, j]),
                "cooc_sum": float(cooc_sum[i, j]),
                "cooc_topk": float(cooc_topk[i, j]),
            })

    return pd.DataFrame(rows)


# ================================
# 🏆 TRAINING RANKER
# ================================
def _train_ranker_ensemble(
    long_df: pd.DataFrame,
    windows: tuple[int, ...],
    params: dict,
    k_eval: int,
    cfg: Config,
):
    models = []
    last_idx = long_df["draw_idx"].max()

    for win in windows:
        start_idx = max(0, last_idx - win + 1)
        dfw = long_df[(long_df["draw_idx"] >= start_idx)].copy()

        feature_cols = [
            c
            for c in dfw.columns
            if c not in ("draw_idx", "date", "entity_id", "label")
        ]
        valid_cols = [
            c for c in feature_cols if not dfw[c].isna().all() and dfw[c].nunique() > 1
        ]

        draw_ids = sorted(dfw["draw_idx"].unique())
        n_draws = len(draw_ids)
        n_val = max(1, int(0.1 * n_draws))
        val_draws = set(draw_ids[-n_val:])

        train_mask = ~dfw["draw_idx"].isin(val_draws)

        Xtr = dfw.loc[train_mask, valid_cols + ["entity_id"]].copy()
        Xtr["entity_id"] = Xtr["entity_id"].astype("category")
        ytr = dfw.loc[train_mask, "label"].astype(int).values
        gtr = dfw.loc[train_mask].groupby("draw_idx", sort=False).size().values
        sw_tr = _sample_weights_by_recency(
            dfw.loc[train_mask, "draw_idx"].values, half_life=win // 2
        )

        params_gpu = _maybe_gpu(params, cfg)
        try:
            model = lgb.LGBMRanker(**params_gpu)
            model.fit(Xtr, ytr, group=gtr, sample_weight=sw_tr)
        except Exception:
            model = lgb.LGBMRanker(**params)
            model.fit(Xtr, ytr, group=gtr, sample_weight=sw_tr)

        models.append(model)

    return models


# ================================
# 🎯 TRAINING CLASSIFIER + OOF CALIBRATION
# ================================
def _train_classifier_oof(
    long_df: pd.DataFrame, params: dict, cfg: Config, n_splits: int
):
    feature_cols = [
        c
        for c in long_df.columns
        if c not in ("draw_idx", "date", "entity_id", "label")
    ]
    valid_cols = [
        c
        for c in feature_cols
        if not long_df[c].isna().all() and long_df[c].nunique() > 1
    ]

    X_all = long_df[valid_cols + ["entity_id"]].copy()
    X_all["entity_id"] = X_all["entity_id"].astype("category")
    y_all = long_df["label"].astype(int).values
    idx_all = long_df["draw_idx"].values

    tss = TimeSeriesSplit(n_splits=n_splits)
    oof_pred = np.zeros_like(y_all, dtype=np.float32)
    oof_mask = np.zeros_like(y_all, dtype=bool)

    unique_draws = np.unique(idx_all)

    for tr_idx, va_idx in tss.split(unique_draws):
        tr_draws = set(unique_draws[tr_idx])
        va_draws = set(unique_draws[va_idx])

        tr_mask = np.isin(idx_all, list(tr_draws))
        va_mask = np.isin(idx_all, list(va_draws))

        Xtr, ytr = X_all.loc[tr_mask], y_all[tr_mask]
        Xva, yva = X_all.loc[va_mask], y_all[va_mask]

        sw_tr = _sample_weights_by_recency(
            long_df.loc[tr_mask, "draw_idx"].values,
            half_life=int(np.median([cfg.half_life_numbers, cfg.half_life_stars])),
        )

        params_gpu = _maybe_gpu(params, cfg)
        clf = lgb.LGBMClassifier(**params_gpu)
        try:
            clf.fit(Xtr, ytr, sample_weight=sw_tr)
        except Exception:
            clf = lgb.LGBMClassifier(**params)
            clf.fit(Xtr, ytr, sample_weight=sw_tr)

        oof_pred[va_mask] = clf.predict_proba(Xva)[:, 1]
        oof_mask[va_mask] = True

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_pred[oof_mask], y_all[oof_mask])

    sw_all = _sample_weights_by_recency(
        idx_all, half_life=int(np.median([cfg.half_life_numbers, cfg.half_life_stars]))
    )

    params_gpu = _maybe_gpu(params, cfg)
    clf_full = lgb.LGBMClassifier(**params_gpu)
    try:
        clf_full.fit(X_all, y_all, sample_weight=sw_all)
    except Exception:
        clf_full = lgb.LGBMClassifier(**params)
        clf_full.fit(X_all, y_all, sample_weight=sw_all)

    return clf_full, iso


# ================================
# 🔮 PREDICTION ENSEMBLE FUSIONNÉE
# ================================
def _prepare_next(long_df: pd.DataFrame) -> pd.DataFrame:
    feat_cols = [
        c
        for c in long_df.columns
        if c not in ("draw_idx", "date", "entity_id", "label")
    ]
    last_idx = int(long_df["draw_idx"].max())
    next_df = long_df[long_df["draw_idx"] == last_idx][["entity_id"] + feat_cols].copy()
    next_df["entity_id"] = next_df["entity_id"].astype("category")
    return next_df


def _predict_next_fused(
    long_df: pd.DataFrame,
    cfg: Config,
    windows: tuple[int, ...],
    ranker_params: dict,
    classifier_params: dict,
    k_eval: int,
    ranker_weight: float,
    classifier_weight: float,
):
    rankers = _train_ranker_ensemble(long_df, windows, ranker_params, k_eval, cfg)
    Xnext = _prepare_next(long_df)

    rank_scores = []
    for r in rankers:
        s = r.predict(Xnext)
        rank_scores.append(_minmax01(s))

    rank_score = np.mean(np.vstack(rank_scores), axis=0)

    n_splits = (
        cfg.oof_splits_numbers
        if k_eval == cfg.eval_at_numbers
        else cfg.oof_splits_stars
    )

    clf_full, iso = _train_classifier_oof(
        long_df, classifier_params, cfg, n_splits=n_splits
    )
    p_raw_lgb = clf_full.predict_proba(Xnext)[:, 1]
    p_cal_lgb = iso.transform(p_raw_lgb)

    feature_cols = [
        c
        for c in long_df.columns
        if c not in ("draw_idx", "date", "entity_id", "label")
    ]
    valid_cols = [
        c
        for c in feature_cols
        if not long_df[c].isna().all() and long_df[c].nunique() > 1
    ]

    X_all_rf = long_df[valid_cols + ["entity_id"]].copy()
    X_all_rf["entity_id"] = X_all_rf["entity_id"].astype(int)
    y_all = long_df["label"].astype(int).values

    rf_clf = RandomForestClassifier(
        n_estimators=150, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    sw_all = _sample_weights_by_recency(
        long_df["draw_idx"].values,
        half_life=int(np.median([cfg.half_life_numbers, cfg.half_life_stars])),
    )
    rf_clf.fit(X_all_rf, y_all, sample_weight=sw_all)

    Xnext_rf = Xnext.copy()
    Xnext_rf["entity_id"] = Xnext_rf["entity_id"].astype(int)
    p_cal_rf = rf_clf.predict_proba(Xnext_rf[valid_cols + ["entity_id"]])[:, 1]

    p_cal = 0.6 * p_cal_lgb + 0.4 * p_cal_rf
    fused = ranker_weight * rank_score + classifier_weight * p_cal

    return (
        pd
        .DataFrame({
            "entity_id": Xnext["entity_id"].astype(int).values,
            "rank_score": rank_score,
            "p_clf_raw": p_raw_lgb,
            "p_clf_cal": p_cal,
            "score_fused": fused,
        })
        .sort_values("entity_id")
        .reset_index(drop=True)
    )


# ================================
# 📊 MÉTRIQUES & SIMULATION FINANCIÈRE
# ================================
def dcg_at_k(rels: list[int], k: int) -> float:
    s = 0.0
    for i, r in enumerate(rels[:k], start=1):
        s += (2**r - 1) / log2(i + 1)
    return s


def ndcg_at_k(y_true_binary: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    order = np.argsort(-y_scores)
    rels = y_true_binary[order].tolist()
    dcg = dcg_at_k(rels, k)

    ideal_rels = sorted(y_true_binary.tolist(), reverse=True)
    idcg = dcg_at_k(ideal_rels, k)

    return 0.0 if idcg == 0 else dcg / idcg


def simulate_financial_payout(
    ticket_nums: list[int],
    ticket_stars: list[int],
    actual_nums: list[int],
    actual_stars: list[int],
) -> float:
    matched_nums = len(set(ticket_nums).intersection(set(actual_nums)))
    matched_stars = len(set(ticket_stars).intersection(set(actual_stars)))

    payouts = {
        (5, 2): 50000000.0,
        (5, 1): 403987.0,
        (5, 0): 43973.0,
        (4, 2): 2067.0,
        (4, 1): 138.0,
        (3, 2): 74.0,
        (4, 0): 46.0,
        (2, 2): 16.0,
        (3, 1): 12.0,
        (3, 0): 10.0,
        (1, 2): 7.88,
        (2, 1): 6.25,
        (2, 0): 4.08,
    }
    return payouts.get((matched_nums, matched_stars), 0.0)


# ================================
# 📈 PLOTS
# ================================
def _plot_line(x, y, title, xlabel, ylabel, path):
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


# ================================
# 🎟️ PORTFOLIO
# ================================
def _mix_sampling_probs(pred_df: pd.DataFrame, alpha: float) -> np.ndarray:
    r = _minmax01(pred_df["rank_score"].values)
    p = alpha * pred_df["p_clf_cal"].values + (1.0 - alpha) * r
    p = np.clip(p, 1e-12, None)
    return p / p.sum()


def _gumbel_top_k(probs: np.ndarray, k: int, rng: np.random.Generator) -> list[int]:
    g = rng.gumbel(size=len(probs))
    keys = np.log(probs) + g
    order = np.argsort(-keys)[:k]
    return (order + 1).tolist()


def is_ticket_physically_coherent(nums: list[int]) -> bool:
    sorted_nums = sorted(nums)
    if (sorted_nums[-1] - sorted_nums[0]) < 22:
        return False
    if sorted_nums[0] > 18:
        return False
    if sorted_nums[-1] < 32:
        return False

    dizaines = [n // 10 if n < 50 else 4 for n in sorted_nums]
    from collections import Counter

    counts = Counter(dizaines)
    return not any(qty >= 3 for qty in counts.values())


def make_portfolio(
    pred_numbers: pd.DataFrame, pred_stars: pd.DataFrame, cfg: Config, seed: int = 123
) -> list[tuple[list[int], list[int]]]:
    rng = np.random.default_rng(seed)

    pn = _mix_sampling_probs(pred_numbers, cfg.sampling_alpha_numbers)
    ps = _mix_sampling_probs(pred_stars, cfg.sampling_alpha_stars)

    tickets = []
    seen = set()
    attempts = 0
    max_filtered_attempts = max(cfg.portfolio_max_attempts, 5000)

    while len(tickets) < cfg.n_tickets and attempts < max_filtered_attempts:
        attempts += 1
        nums = _gumbel_top_k(pn, cfg.eval_at_numbers, rng)
        if is_ticket_physically_coherent(nums):
            stars = _gumbel_top_k(ps, cfg.eval_at_stars, rng)
            nums.sort()
            stars.sort()
            key = (tuple(nums), tuple(stars))
            if key not in seen:
                seen.add(key)
                tickets.append((nums, stars))

    if len(tickets) < cfg.n_tickets:
        while len(tickets) < cfg.n_tickets:
            nums = sorted(_gumbel_top_k(pn, cfg.eval_at_numbers, rng))
            stars = sorted(_gumbel_top_k(ps, cfg.eval_at_stars, rng))
            key = (tuple(nums), tuple(stars))
            if key not in seen:
                seen.add(key)
                tickets.append((nums, stars))

    return tickets


# ================================
# 🔍 BACKTEST + SIMULATION FINANCIÈRE
# ================================
def backtest(df_draws: pd.DataFrame, cfg: Config, out_dir: str):
    n_splits_backtest = getattr(cfg, "n_splits_backtest", 20)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("🚀 Pré-calcul des tables de features (avec Gaps Avancés)...")

    dataset_hash = cache.hash_dataset(df_draws)
    params_hash = cache.hash_params(cfg)
    code_hash = cache.hash_code(build_long_table)

    cache_key_num = cache.make_cache_key(
        dataset_hash, params_hash, code_hash, "long_numbers_v2"
    )
    cache_key_star = cache.make_cache_key(
        dataset_hash, params_hash, code_hash, "long_stars_v2"
    )

    long_num = cache.load(cache_key_num)
    long_star = cache.load(cache_key_star)

    if long_num is None or long_star is None:
        print("⚙️ Calcul des tables longues...")
        long_num = build_long_table(df_draws, cfg.pool_numbers, "number", cfg)
        long_star = build_long_table(df_draws, cfg.pool_stars, "star", cfg)

        cache.save(cache_key_num, long_num)
        cache.save(cache_key_star, long_star)

    last_idx = int(min(long_num["draw_idx"].max(), long_star["draw_idx"].max()))
    split_points = np.linspace(
        cfg.min_history_draws, last_idx - 1, n_splits_backtest, dtype=int
    )

    rows = []
    total_spent = 0.0
    total_winnings = 0.0

    print("\n📊 Lancement du Backtesting (NDCG + Simulation Financière)...")

    for sp in tqdm(split_points, desc="Backtest", unit="split"):
        tr_num = long_num[long_num["draw_idx"] <= sp].copy()
        te_num = long_num[long_num["draw_idx"] == sp + 1].copy()

        tr_star = long_star[long_star["draw_idx"] <= sp].copy()
        te_star = long_star[long_star["draw_idx"] == sp + 1].copy()

        if te_num.empty or te_star.empty:
            continue

        pred_num = _predict_next_fused(
            tr_num,
            cfg,
            cfg.windows_numbers,
            cfg.lgb_ranker_params_numbers,
            cfg.lgb_classifier_params_numbers,
            cfg.eval_at_numbers,
            cfg.ranker_weight_numbers,
            cfg.classifier_weight_numbers,
        )

        pred_star = _predict_next_fused(
            tr_star,
            cfg,
            cfg.windows_stars,
            cfg.lgb_ranker_params_stars,
            cfg.lgb_classifier_params_stars,
            cfg.eval_at_stars,
            cfg.ranker_weight_stars,
            cfg.classifier_weight_stars,
        )

        y_true_num = te_num.sort_values("entity_id")["label"].values
        y_true_star = te_star.sort_values("entity_id")["label"].values

        y_score_num = pred_num.sort_values("entity_id")["score_fused"].values
        y_score_star = pred_star.sort_values("entity_id")["score_fused"].values

        ndcg_num = ndcg_at_k(y_true_num, y_score_num, cfg.eval_at_numbers)
        ndcg_star = ndcg_at_k(y_true_star, y_score_star, cfg.eval_at_stars)

        actual_winning_nums = te_num[te_num["label"] == 1]["entity_id"].tolist()
        actual_winning_stars = te_star[te_star["label"] == 1]["entity_id"].tolist()

        portfolio_tickets = make_portfolio(pred_num, pred_star, cfg, seed=int(sp))
        split_winnings = 0.0
        for t_nums, t_stars in portfolio_tickets:
            total_spent += 2.50
            split_winnings += simulate_financial_payout(
                t_nums, t_stars, actual_winning_nums, actual_winning_stars
            )

        total_winnings += split_winnings

        rows.append({
            "train_upto_idx": int(sp),
            "test_idx": int(sp + 1),
            "ndcg_numbers": float(ndcg_num),
            "ndcg_stars": float(ndcg_star),
            "split_winnings_eur": float(split_winnings),
        })

        del pred_num, pred_star, tr_num, te_num, tr_star, te_star
        gc.collect()

    res = pd.DataFrame(rows)

    print(f"\n💰 Bilan Financier Virtuel du Backtest :")
    print(
        f"   - Total dépensé ({int(total_spent / 2.5)} grilles) : {total_spent:.2f} €"
    )
    print(f"   - Total gains simulés : {total_winnings:.2f} €")
    print(f"   - Bilan Net : {total_winnings - total_spent:.2f} €\n")

    if out:
        res.to_csv(out / "backtest_results.csv", index=False)
        try:
            _plot_line(
                res["test_idx"].values,
                res["ndcg_numbers"].values,
                "NDCG@5 — Numéros (Gaps & Ensemble)",
                "Index test",
                "NDCG@5",
                out / "ndcg_numbers.png",
            )
            _plot_line(
                res["test_idx"].values,
                res["ndcg_stars"].values,
                "NDCG@2 — Étoiles (Gaps & Ensemble)",
                "Index test",
                "NDCG@2",
                out / "ndcg_stars.png",
            )
        except Exception as e:
            print(f"⚠️ Impossible de générer les PNG : {e}")

    return res


# ================================
# 🚀 TRAIN + PREDICT FINAL
# ================================
def train_and_predict_for_next(df_draws: pd.DataFrame, cfg: Config):
    dataset_hash = cache.hash_dataset(df_draws)
    params_hash = cache.hash_params(cfg)
    code_hash = cache.hash_code(build_long_table)

    cache_key_num = cache.make_cache_key(
        dataset_hash, params_hash, code_hash, "long_numbers_v2"
    )
    cache_key_star = cache.make_cache_key(
        dataset_hash, params_hash, code_hash, "long_stars_v2"
    )

    long_num = cache.load(cache_key_num)
    long_star = cache.load(cache_key_star)

    if long_num is None or long_star is None:
        long_num = build_long_table(df_draws, cfg.pool_numbers, "number", cfg)
        long_star = build_long_table(df_draws, cfg.pool_stars, "star", cfg)
        cache.save(cache_key_num, long_num)
        cache.save(cache_key_star, long_star)

    pred_num = _predict_next_fused(
        long_num,
        cfg,
        cfg.windows_numbers,
        cfg.lgb_ranker_params_numbers,
        cfg.lgb_classifier_params_numbers,
        cfg.eval_at_numbers,
        cfg.ranker_weight_numbers,
        cfg.classifier_weight_numbers,
    )
    pick_numbers = (
        pred_num
        .sort_values("score_fused", ascending=False)
        .head(cfg.eval_at_numbers)["entity_id"]
        .tolist()
    )

    pred_star = _predict_next_fused(
        long_star,
        cfg,
        cfg.windows_stars,
        cfg.lgb_ranker_params_stars,
        cfg.lgb_classifier_params_stars,
        cfg.eval_at_stars,
        cfg.ranker_weight_stars,
        cfg.classifier_weight_stars,
    )
    pick_stars = (
        pred_star
        .sort_values("score_fused", ascending=False)
        .head(cfg.eval_at_stars)["entity_id"]
        .tolist()
    )

    return {
        "pred_numbers": pred_num,
        "pred_stars": pred_star,
        "pick_numbers": pick_numbers,
        "pick_stars": pick_stars,
    }


# ================================
# 🧪 POINT D'ENTRÉE DU PIPELINE
# ================================
def run_pipeline(csv_path: str, out_dir: str, cfg: Config) -> dict:
    df = load_draws(csv_path)

    print("\n📊 Backtest en cours...")
    backtest(df, cfg, out_dir=out_dir)

    print("\n🔮 Entraînement final sur l'historique complet...")
    out_pred = train_and_predict_for_next(df, cfg)

    pred_num = out_pred["pred_numbers"]
    pred_star = out_pred["pred_stars"]

    top5_numbers = out_pred["pick_numbers"]
    top2_stars = out_pred["pick_stars"]

    portfolio = make_portfolio(pred_num, pred_star, cfg, seed=123)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pred_num.to_csv(out_path / "pred_numbers_next.csv", index=False)
    pred_star.to_csv(out_path / "pred_stars_next.csv", index=False)

    with open(out_path / "portfolio_7_tickets.txt", "w") as f:
        for i, (nums, stars) in enumerate(portfolio, 1):
            f.write(f"Ticket {i:02d}: N {nums} | S {stars}\n")

    port_rows = []
    for i, (nums, stars) in enumerate(portfolio, 1):
        row = {"ticket": i}
        for k, v in enumerate(nums, 1):
            row[f"n{k}"] = v
        for k, v in enumerate(stars, 1):
            row[f"s{k}"] = v
        port_rows.append(row)

    portfolio_csv = out_path / "portfolio_7_tickets.csv"
    pd.DataFrame(port_rows).to_csv(portfolio_csv, index=False)

    return {
        "top5_numbers": top5_numbers,
        "top2_stars": top2_stars,
        "portfolio_csv": str(portfolio_csv),
    }


if __name__ == "__main__":
    cfg = Config()
    run_pipeline("euromillions.csv", "resultats_euromillions", cfg)
