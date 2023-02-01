from typing import List, Optional

import pandas as pd


##
def data_check(
    data: pd.DataFrame,
    response_rates: str = None,
    response: str = None,
    additional_args: List[Optional[str]] = None,
) -> pd.DataFrame:
    """
    Convert data into correct format before passing it to the hssm models

    data: data should be in pandas format
    response_rates: name of the column indicating response rates
    response: name of the column indicating response rates
    additional_args: list of additional columns that will be used in the model
    """
    if all(v is None for v in [response_rates, response, additional_args]):
        columns = ["rt", "response"]
    elif additional_args is None:
        columns = [response_rates, response]
    elif additional_args is not None:
        columns = [response_rates, response] + additional_args
    data = data[columns]
    return data
