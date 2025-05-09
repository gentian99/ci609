from preprocessing import (
    load_and_filter_data,
    handle_missing_values,
    feature_engineering_prematch,
    scale_features_and_split
)

def main():
    #load raw match data from CSVs
    df_raw = load_and_filter_data()
    print("Raw data shape:", df_raw.shape)

    #fill missing values in both numeric and object columns
    df_clean = handle_missing_values(df_raw)
    print("After missing value handling:", df_clean.shape)

    #create features
    df_features = feature_engineering_prematch(df_clean, encode_teams=True)
    print("After feature engineering:", df_features.shape)

    #encode target labels (FTR → FTR_Code)
    label_map = {'H': 0, 'D': 1, 'A': 2}
    df_clean['FTR_Code'] = df_clean['FTR'].map(label_map)
    df_features['FTR_Code'] = df_clean['FTR_Code']

    #scale numeric features and split into training/test sets
    X_train, X_test, y_train, y_test, _ = scale_features_and_split(
        df_features, target_column='FTR_Code'
    )
    print("Training set:", X_train.shape, "| Test set:", X_test.shape)

if __name__ == '__main__':
    main()
