def add_time_features(df):


    df["hour_of_day"]=(

        df["purchase_time"]

        .dt.hour

    )


    return df
