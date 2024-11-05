import pandas as pd

from typing import Tuple, List

from functions import iqr_range_target_filter, compare_feature_lists

def test_iqr_range_target_filter():
    # Create a DataFrame with some outliers
    df = pd.DataFrame({
        'price': [100, 150, 200, 250, 300, 1000]  # 1000 is an outlier
    })

    # Call the function
    filtered_df = iqr_range_target_filter(df, 'price')

    # Assert that the outlier (1000) is removed
    assert 1000 not in filtered_df['price'].values # 1000 est un outlier qui n'a pas été retiré

    # Assert that the number of rows is less than the original due to outlier removal
    assert filtered_df.shape[0] < df.shape[0]

    # Create a DataFrame without outliers
    df_no_outliers = pd.DataFrame({
        'price': [100, 150, 200, 250, 300]
    })

    # Call the function again
    filtered_df_no_outliers = iqr_range_target_filter(df_no_outliers, 'price')

    # Assert that the number of rows remains the same
    assert filtered_df_no_outliers.shape[0] == df_no_outliers.shape[0]


def test_compare_feature_lists():
        # Example usage:
    list1 = ['feature1', 'feature2', 'feature3', 'feature4']
    list2 = ['feature3', 'feature4', 'feature5', 'feature6']

    common, unique = compare_feature_lists(list1, list2)

    assert common ==  ['feature4', 'feature3'] # features in common aren't properly returned
    assert unique ==  ['feature1', 'feature2', 'feature6', 'feature5'] # unique features aren't properly returned
