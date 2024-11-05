# price_column is a str name of the numeric column
def iqr_range_target_filter(df, price_column):
    q3 = df.price.describe()['75%']
    q1 = df.price.describe()['25%']
    iqr_range = q3-q1
    new_df = df[(df[price_column] > q1-iqr_range*1.5) & (df[price_column] < q3+(iqr_range*1.5))]
    print("old number of rows", df.shape[0])
    print("new number of rows", new_df.shape[0])
    return new_df
