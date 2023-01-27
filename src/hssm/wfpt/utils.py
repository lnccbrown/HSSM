from typing import List

import pandas as pd


def data_check(
    data: pd.DataFrame,
    response_rates: str,
    response: str,
    additional_args: List[str] = None,
) -> pd.DataFrame:
    """
    Convert data into correct format before passing it to the hssm models

    data: data should be in pandas format
    response_rates: name of the column indicating response rates
    response: name of the column indicating response rates
    additional_args: list of additional columns that will be used in the model
    """
    if additional_args is None:
        columns = [response_rates, response]
    else:
        columns = [response_rates, response] + additional_args
    data = data[columns]
    return data
