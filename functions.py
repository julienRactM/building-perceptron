# price_column is a str name of the numeric column
def iqr_range_target_filter(df, price_column):
    q3 = df.price.describe()['75%']
    q1 = df.price.describe()['25%']
    iqr_range = q3-q1
    new_df = df[(df[price_column] > q1-iqr_range*1.5) & (df[price_column] < q3+(iqr_range*1.5))]
    print("old number of rows", df.shape[0])
    print("new number of rows", new_df.shape[0])
    return new_df


def compare_feature_lists(list1, list2):
    """
    # Example usage:
    list1 = ['feature1', 'feature2', 'feature3', 'feature4']
    list2 = ['feature3', 'feature4', 'feature5', 'feature6']

    common, unique = compare_feature_lists(list1, list2)
    print("Common Elements:", common)
    print("Unique Elements:", unique)

    renvoie

    Common Elements: ['feature4', 'feature3']
    Unique Elements: ['feature1', 'feature2', 'feature6', 'feature5']

    """
    # Find common elements (present in both lists)
    common_elements = list(set(list1) & set(list2))

    # Find unique elements (present in only one of the lists)
    unique_elements = list(set(list1) ^ set(list2))

    return common_elements, unique_elements
