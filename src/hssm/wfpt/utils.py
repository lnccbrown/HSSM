from typing import List

import pandas as pd


##
def data_check(
    data: pd.DataFrame,
    response_rates: str = None,
    response: str = None,
    additional_args: List = None,
) -> pd.DataFrame:
    """
    Convert data into correct format before passing it to the hssm models

    data: data should be in pandas format
    response_rates: name of the column indicating response rates
    response: name of the column indicating response rates
    additional_args: list of additional columns that will be used in the model
    """
    replace_dict = {
        ("rt", "response", None): ["rt", "response"],
        (None, None, None): ["rt", "response"],
        (None, None, *additional_args): ["rt", "response", *additional_args],
    }
    new_columns = replace_dict[tuple([response_rates, response, *additional_args])]
    data = data[new_columns]
    return data
