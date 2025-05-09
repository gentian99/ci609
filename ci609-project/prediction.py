import glob
import re
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from preprocessing import (
    load_and_filter_data,
    handle_missing_values,
    feature_engineering_cumulative,
    generate_sequence_data
)

#get most recent stats for a team role (home or away)
def get_recent_cumulative_stats(df, team, role):
    key = 'HomeTeam' if role.lower() == 'home' else 'AwayTeam'
    team_df = df[df[key] == team]
    if team_df.empty:
        return {}

    recent = team_df.sort_values('Date').iloc[-1]
    keys = [
        f'{role}_cum_avg_for', f'{role}_cum_avg_against',
        f'{role}_cum_win_rate', f'{role}_roll_avg_for',
        f'{role}_roll_avg_against', f'{role}_roll_win_rate',
        f'{role}_conversion_rate'
    ]
    return {k: recent.get(k, 0) for k in keys}

#assemble input features for the tabular model
def prepare_match_features(home_team, away_team):
    df = load_and_filter_data()
    df = df[df['Season'] == '2024_2025'].copy()
    df = handle_missing_values(df)
    df = feature_engineering_cumulative(df, encode_teams=False)

    home_stats = get_recent_cumulative_stats(df, home_team, 'home')
    away_stats = get_recent_cumulative_stats(df, away_team, 'away')

    features = {**home_stats, **away_stats}
    return pd.DataFrame([features])

#prepare LSTM sequence input
def prepare_sequence_input(home_team, away_team, window_size=5):
    df = load_and_filter_data()
    df = df[df['Season'] == '2024_2025'].copy()
    df = handle_missing_values(df)
    df = feature_engineering_cumulative(df, encode_teams=False)
    sequences = generate_sequence_data(df, window_size=window_size)

    idxs = df[(df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)].index
    idx = idxs[-1] if len(idxs) else df[df['HomeTeam'] == home_team].index[-1]

    return np.expand_dims(sequences[idx], axis=0)

#main prediction function
def predict_match(home_team, away_team):
    model_files = sorted(
        glob.glob("football_hybrid_model_fold_*.h5"),
        key=lambda f: int(re.search(r"fold_(\d+)", f).group(1))
    )
    models = [load_model(f) for f in model_files]

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)

    X_tab = prepare_match_features(home_team, away_team).reindex(columns=feature_columns, fill_value=0)
    X_tab_scaled = scaler.transform(X_tab)
    X_seq = prepare_sequence_input(home_team, away_team)

    predictions = [model.predict([X_tab_scaled, X_seq])[0] for model in models]
    avg_pred = np.mean(predictions, axis=0)

    # 🧠 Bias correction: boost Home Win if Away confidence is similar
    home_win_boost = 0.05
    if abs(avg_pred[0] - avg_pred[2]) < 0.10:
        avg_pred[0] += home_win_boost
        avg_pred = avg_pred / np.sum(avg_pred)  # re-normalize

    result_class = np.argmax(avg_pred)
    label_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
    return label_map[result_class], avg_pred


    # 3. Prepare inputs
    X_tab = prepare_match_features(home_team, away_team).reindex(columns=feature_columns, fill_value=0)
    X_tab_scaled = scaler.transform(X_tab)
    X_seq = prepare_sequence_input(home_team, away_team)

    # 4. Get predictions from each model and average
    predictions = [model.predict([X_tab_scaled, X_seq])[0] for model in models]
    avg_pred = np.mean(predictions, axis=0)

    # 5. Return result
    result_class = np.argmax(avg_pred)
    label_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
    return label_map[result_class], avg_pred

# --- Optional CLI test ---
if __name__ == '__main__':
    sample_matches = [("Arsenal", "Chelsea"), ("Liverpool", "Man U"), ("Everton", "Leeds")]
    for home, away in sample_matches:
        result, probs = predict_match(home, away)
        print(f"{home} vs {away} → {result} ({probs})")
