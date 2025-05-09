import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_filter_data():
    #core training datasets (Premier League only)
    training_urls = {
        '2022_2023': 'https://www.football-data.co.uk/mmz4281/2223/E0.csv',
        '2023_2024': 'https://www.football-data.co.uk/mmz4281/2324/E0.csv',
        '2024_2025': 'https://www.football-data.co.uk/mmz4281/2425/E0.csv'
    }

    #extra placeholder leagues (not used in training or prediction)
    placeholder_urls = {
        'bundesliga_24_25': 'https://www.football-data.co.uk/mmz4281/2425/D1.csv',
        'serie_a_24_25': 'https://www.football-data.co.uk/mmz4281/2425/I1.csv',
        'la_liga_24_25': 'https://www.football-data.co.uk/mmz4281/2425/SP1.csv',
        'ligue_1_24_25': 'https://www.football-data.co.uk/mmz4281/2425/F1.csv'
    }

    desired_columns = [
        'Div', 'Date', 'Time', 'HomeTeam', 'AwayTeam',
        'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',
        'Referee', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF',
        'HC', 'AC', 'HY', 'AY', 'HR', 'AR'
    ]

    dfs = []

    for season, url in training_urls.items():
        try:
            df = pd.read_csv(url)
            df['Season'] = season
            common_cols = [col for col in desired_columns if col in df.columns]
            df_filtered = df[common_cols + ['Season']]
            dfs.append(df_filtered)
            print(f"{season} loaded with columns: {common_cols}")
        except Exception as e:
            print(f"Error loading data for {season}: {e}")

    for season, url in placeholder_urls.items():
        try:
            df = pd.read_csv(url)
            df['Season'] = season
            placeholder_df = df[['HomeTeam', 'AwayTeam', 'Date']].copy()
            placeholder_df['FTHG'] = 0
            placeholder_df['FTAG'] = 0
            placeholder_df['FTR'] = 'D'
            placeholder_df['Season'] = season
            dfs.append(placeholder_df)
            print(f"{season} placeholder loaded with teams only.")
        except Exception as e:
            print(f"Error loading placeholder league {season}: {e}")

    df_raw = pd.concat(dfs, ignore_index=True)
    return df_raw

def handle_missing_values(df):
    df_imputed = df.copy()
    numeric_cols = df_imputed.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())
    object_cols = df_imputed.select_dtypes(include=['object']).columns
    for col in object_cols:
        df_imputed[col] = df_imputed[col].ffill()
    return df_imputed

def feature_engineering_cumulative(df, encode_teams=False):
    df_feat = df.copy()
    df_feat['Date'] = pd.to_datetime(df_feat['Date'], errors='coerce', dayfirst=True)
    df_feat = df_feat.sort_values('Date').reset_index(drop=True)

    df_feat['home_cum_avg_for'] = df_feat.groupby('HomeTeam')['FTHG'].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean())
    df_feat['home_cum_avg_against'] = df_feat.groupby('HomeTeam')['FTAG'].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean())
    df_feat['away_cum_avg_for'] = df_feat.groupby('AwayTeam')['FTAG'].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean())
    df_feat['away_cum_avg_against'] = df_feat.groupby('AwayTeam')['FTHG'].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean())

    df_feat['home_win'] = (df_feat['FTR'] == 'H').astype(int)
    df_feat['away_win'] = (df_feat['FTR'] == 'A').astype(int)
    df_feat['home_cum_win_rate'] = df_feat.groupby('HomeTeam')['home_win'].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean())
    df_feat['away_cum_win_rate'] = df_feat.groupby('AwayTeam')['away_win'].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean())

    window_size = 3
    df_feat['home_roll_avg_for'] = df_feat.groupby('HomeTeam')['FTHG'].transform(
        lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean())
    df_feat['home_roll_avg_against'] = df_feat.groupby('HomeTeam')['FTAG'].transform(
        lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean())
    df_feat['away_roll_avg_for'] = df_feat.groupby('AwayTeam')['FTAG'].transform(
        lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean())
    df_feat['away_roll_avg_against'] = df_feat.groupby('AwayTeam')['FTHG'].transform(
        lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean())

    df_feat['home_roll_win_rate'] = df_feat.groupby('HomeTeam')['home_win'].transform(
        lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean())
    df_feat['away_roll_win_rate'] = df_feat.groupby('AwayTeam')['away_win'].transform(
        lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean())

    df_feat['home_conversion_rate'] = np.where(df_feat['HS'] > 0, df_feat['HST'] / df_feat['HS'], 0)
    df_feat['away_conversion_rate'] = np.where(df_feat['AS'] > 0, df_feat['AST'] / df_feat['AS'], 0)
    df_feat['ht_goal_diff'] = df_feat['HTHG'] - df_feat['HTAG']

    num_cols = df_feat.select_dtypes(include=[np.number]).columns
    df_feat[num_cols] = df_feat[num_cols].fillna(df_feat[num_cols].mean())

    if encode_teams:
        df_feat = pd.get_dummies(df_feat, columns=['HomeTeam', 'AwayTeam'], drop_first=False)

    drop_cols = ['FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'home_win', 'away_win']
    df_feat = df_feat.drop(columns=[col for col in drop_cols if col in df_feat.columns])

    return df_feat

def scale_features_and_split(df, target_column='FTR_Code'):
    df_numeric = df.select_dtypes(include=[np.number])
    X_full = df_numeric.drop([target_column], axis=1)
    y = df_numeric[target_column]
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_full), columns=X_full.columns, index=X_full.index)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler

def generate_sequence_data(df, window_size=5, feature_names=None):
    if feature_names is None:
        feature_names = ['home_roll_avg_for', 'home_roll_avg_against', 'home_roll_win_rate', 'home_conversion_rate',
                         'away_roll_avg_for', 'away_roll_avg_against', 'away_roll_win_rate', 'away_conversion_rate']

    sequences = []
    for i, row in df.iterrows():
        current_date = row['Date']
        home_team = row.get('HomeTeam')
        away_team = row.get('AwayTeam')

        seq_data = np.zeros((window_size, len(feature_names)))

        home_matches = df[(df['HomeTeam'] == home_team) & (df['Date'] < current_date)].sort_values('Date').tail(window_size)
        away_matches = df[(df['AwayTeam'] == away_team) & (df['Date'] < current_date)].sort_values('Date').tail(window_size)

        home_seq = home_matches[feature_names].mean(axis=0) if not home_matches.empty else np.zeros(len(feature_names))
        away_seq = away_matches[feature_names].mean(axis=0) if not away_matches.empty else np.zeros(len(feature_names))

        combined = np.concatenate([home_seq, away_seq])
        seq = np.tile(combined, (window_size, 1))
        sequences.append(seq)

    return np.array(sequences)

if __name__ == '__main__':
    df_raw = load_and_filter_data()
    print("Raw data shape:", df_raw.shape)
    df_clean = handle_missing_values(df_raw)
    print("After missing value handling:", df_clean.shape)
    df_feat = feature_engineering_cumulative(df_clean, encode_teams=False)
    print("After feature engineering:", df_feat.shape)

    mapping = {'H': 0, 'D': 1, 'A': 2}
    df_clean['FTR_Code'] = df_clean['FTR'].map(mapping)
    df_feat['FTR_Code'] = df_clean['FTR_Code']
    X_train, X_test, y_train, y_test, scaler = scale_features_and_split(df_feat)
    print("Training set shape:", X_train.shape, "Test set shape:", X_test.shape)

    seq_data = generate_sequence_data(df_feat)
    print("Sequence data shape (for all samples):", seq_data.shape)
    print(df_raw.head())
