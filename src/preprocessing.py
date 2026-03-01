import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from .config import LABEL_WINDOW_DAYS, MIN_GAMES_PER_PLAYER, SPLIT_DATE, SEED

def _make_safe_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()

    safe_cols = []
    used = set()

    for c in df.columns:
        new = re.sub(r"[^0-9a-zA-Z_]+", "_", c)
        orig_new = new
        k = 1
        while new in used:
            new = f"{orig_new}_{k}"
            k += 1
        used.add(new)
        safe_cols.append(new)

    rename_map = dict(zip(df.columns, safe_cols))
    return df.rename(columns=rename_map)

def add_position_multilabel(df: pd.DataFrame) -> pd.DataFrame:
    
    if "position" not in df.columns:
        return df

    out = df.copy()
    out["position_list"] = out["position"].apply(
        lambda x: x.split("/") if isinstance(x, str) and x != "unknown" else []
    )

    mlb = MultiLabelBinarizer()
    raw_binary = mlb.fit_transform(out["position_list"])
    position_label_cols = [f"position_{c}" for c in mlb.classes_]

    position_binary = pd.DataFrame(raw_binary, columns=position_label_cols, index=out.index)
    out = pd.concat([out, position_binary], axis=1)
    return out

def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    
    out = df.copy()
    out["gameDate_player"] = pd.to_datetime(out["gameDate_player"])
    if "injury_date" in out.columns:
        out["injury_date"] = pd.to_datetime(out["injury_date"], errors="coerce")

    out = out.sort_values(["personId", "gameDate_player"]).reset_index(drop=True)

    out["injury_indicator"] = (
        out.groupby("personId")["injury_date"]
           .transform(lambda s: s.ne(s.shift()) & s.notna())
    ).astype(int)
    
    out["injury_probability"] = 0

    for pid in out.loc[out["injury_indicator"] == 1, "personId"].unique():
        event_dates = (
            out.loc[(out["personId"] == pid) & (out["injury_indicator"] == 1), "gameDate_player"]
               .dropna()
               .unique()
        )
        for event_date in event_dates:
            event_date = pd.to_datetime(event_date)
            window_start = event_date - pd.Timedelta(days=LABEL_WINDOW_DAYS)
            mask = (
                (out["personId"] == pid)
                & (out["gameDate_player"] >= window_start)
                & (out["gameDate_player"] < event_date)
            )
            out.loc[mask, "injury_probability"] = 1

    return out

def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    
    out = df.copy()
    
    if "minutes" in out.columns:
        out["minutes_shift"] = out.groupby("personId")["minutes"].shift(1)

        out["acute_workload"] = (
            out.groupby("personId")["minutes_shift"].rolling(3).mean().reset_index(level=0, drop=True)
        )
        out["chronic_workload"] = (
            out.groupby("personId")["minutes_shift"].rolling(15).mean().reset_index(level=0, drop=True)
        )
        out["acwr"] = out["acute_workload"] / (out["chronic_workload"] + 1e-6)

        out["workload_mean_10"] = (
            out.groupby("personId")["minutes_shift"].rolling(10).mean().reset_index(level=0, drop=True)
        )
        out["workload_std_10"] = (
            out.groupby("personId")["minutes_shift"].rolling(10).std().reset_index(level=0, drop=True)
        )
        out["workload_zscore"] = (out["minutes_shift"] - out["workload_mean_10"]) / (out["workload_std_10"] + 1e-6)

        out["minutes_delta_1"] = out.groupby("personId")["minutes_shift"].diff()
        out["minutes_delta_from_acute"] = out["minutes_shift"] - out["acute_workload"]
    
    out["days_since_last_game"] = out.groupby("personId")["gameDate_player"].diff().dt.days
    out["is_b2b"] = (out["days_since_last_game"] == 1).astype(int)

    if "minutes_shift" in out.columns:
        out["games_last_7"] = (
            out.groupby("personId")["minutes_shift"].rolling(7).count().reset_index(level=0, drop=True)
        )
    
    out["recent_injury"] = out.groupby("personId")["injury_indicator"].shift(1)
    out["recent_injury_10"] = (
        out.groupby("personId")["recent_injury"].rolling(10).sum().reset_index(level=0, drop=True)
    )
    
    if "pra" in out.columns:
        out["pra_shift"] = out.groupby("personId")["pra"].shift(1)
        out["pra_trend_5"] = out.groupby("personId")["pra_shift"].rolling(5).mean().reset_index(level=0, drop=True)
        out["pra_change"] = out["pra_shift"] - out["pra_trend_5"]
    
    if "usage_rate" in out.columns:
        out["usage_shift"] = out.groupby("personId")["usage_rate"].shift(1)
        out["usage_volatility"] = (
            out.groupby("personId")["usage_shift"].rolling(8).std().reset_index(level=0, drop=True)
        )
    
    if "age" in out.columns and "minutes_shift" in out.columns:
        out["age_minutes_interaction"] = out["age"] * out["minutes_shift"]
    
    if "experience" in out.columns:
        out["is_rookie"] = (out["experience"] <= 1).astype(int)

    return out

def build_feature_list(df: pd.DataFrame) -> list[str]:
    
    non_predictive_cols = [
        "personId", "player_name", "gameId", "season", "season_game",
        "playerteamCity", "playerteamName", "opponentteamCity", "opponentteamName",
        "position", "position_list", "center", "forward", "guard",
        "Notes", "last_injury_event_date",
    ]

    forbidden_cols = {
        "injury_probability",
        "injury_indicator",
        "injury_event",
        "injury_date",
        "injury_type",
        "injury_type_simple",
        "inj_lower",
        "last_injury_event_date",
    }

    for col in df.columns:
        if "injury" in col.lower():
            forbidden_cols.add(col)
    
    safe_injury_history = {"recent_injury", "recent_injury_10"}
    for c in safe_injury_history:
        forbidden_cols.discard(c)

    df2 = df.loc[:, ~df.columns.duplicated()].copy()
    numeric_bool_cols = df2.select_dtypes(include=["number", "bool"]).columns

    predictive_features = [
        c for c in numeric_bool_cols
        if c not in non_predictive_cols and c not in forbidden_cols
    ]
    
    seen = set()
    unique_features = []
    for c in predictive_features:
        if c not in seen:
            unique_features.append(c)
            seen.add(c)
    
    leakage_suspects = [c for c in unique_features if "injury" in c.lower() and c not in safe_injury_history]
    if leakage_suspects:
        raise ValueError(f"Leakage suspects found in features: {leakage_suspects}")

    return unique_features

def prepare_train_test(df: pd.DataFrame):
    
    out = _make_safe_column_names(df)
    out = add_position_multilabel(out)

    out = build_labels(out)
    out = add_feature_engineering(out)

    out = out.loc[:, ~out.columns.duplicated()].copy()

    features = build_feature_list(out)   

    out = out.dropna(subset=features).copy()
   
    game_counts = out.groupby("personId")["gameDate_player"].transform("count")
    out = out[game_counts >= MIN_GAMES_PER_PLAYER].copy()

    split_date = pd.to_datetime(SPLIT_DATE)
    train_mask = out["gameDate_player"] < split_date
    test_mask = ~train_mask

    X_train = out.loc[train_mask, features]
    X_test = out.loc[test_mask, features]
    y_train = out.loc[train_mask, "injury_probability"].astype(int)
    y_test = out.loc[test_mask, "injury_probability"].astype(int)

    return out, features, (X_train, y_train, X_test, y_test, train_mask, test_mask)