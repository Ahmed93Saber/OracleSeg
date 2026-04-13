from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_and_scale_data_extern(df_internal, df_external, label_col, feature_cols,
                                test_size=0.2, random_state=0, categorical_cols=None):
    """
    Splits the dataframe into train and test sets and scales numerical feature columns.
    Categorical columns are added back without scaling.
    """

    # Train/test split
    train_df, test_df = train_test_split(
        df_internal,
        test_size=test_size,
        random_state=random_state,
        stratify=df_internal[label_col]
    )

    # Scale only the numerical feature columns
    scaler = StandardScaler()
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()
    df_external_scaled = df_external.copy()

    train_df_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
    df_external_scaled[feature_cols] = scaler.transform(df_external[feature_cols])

    train_df_scaled[['num_weeks']] = scaler.fit_transform(train_df[['num_weeks']])
    test_df_scaled[['num_weeks']]= scaler.transform(test_df[['num_weeks']])



    # If categorical columns exist, just carry them over without scaling
    if categorical_cols:
        for col in categorical_cols:
            train_df_scaled[col] = train_df[col].values
            test_df_scaled[col] = test_df[col].values
            df_external_scaled[col] = df_external[col].values

    return train_df_scaled, test_df_scaled, df_external_scaled
