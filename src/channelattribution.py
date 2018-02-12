import pandas as pd
import numpy as np
from channelattribution._libc import heuristic_model, markov_model

#def test_df(df):
#    #Check that df id DataFrame
#    assert isinstance(df, pd.DataFrame), "Provide a pandas DataFrame"
#    #Check that it has a (not necessarily unique) column named double
#    assert 'double' in df.columns, "df has no column 'double'" 
#    #Check type of 'double' column is double
#    assert df.double.dtype is np.dtype('float64'), "'double' columns must be of type float64"
#
#    vec = df.double.values
#    return test(vec) 
