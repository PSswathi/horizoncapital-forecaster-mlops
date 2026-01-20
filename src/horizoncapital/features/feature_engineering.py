def add_lags(df, cols, lags=(1,3,6,12)):
    df = df.copy()
    for c in cols:
        for lag in lags:
            df[f"{c}_lag{lag}"] = df[c].shift(lag)
    return df

def add_rolling(df, cols, windows=(3,6,12)):
    df = df.copy()
    for c in cols:
        for w in windows:
            df[f"{c}_rollmean{w}"] = df[c].rolling(w).mean()
            df[f"{c}_rollstd{w}"] = df[c].rolling(w).std()
    return df
