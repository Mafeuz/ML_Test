import random
import pandas as pd
import numpy as np
from pathlib import Path
from data_utils import retrieve_data_from_pickle, fit_TSNE, plot_TSNE

""" Test Methods for data_utils module.
"""

# Pickle file:
TEST_PICKLE_FILE_PATH = f'{Path(__file__).parent.parent}/mini_gm_public_v0.1.p'

def test_retrieve_data_from_pickle() -> bool:

    """ Method to test the retrivieng of data dict from pickle file.
    """

    data = retrieve_data_from_pickle(TEST_PICKLE_FILE_PATH )
    assert isinstance(data, dict)

def test_fit_TSNE() -> bool:

    """ Method to test TSNE data fit.
    """

    # TSNE parameters:
    n_components = 2
    perplexity = 30

    # Instance of random data:
    random_data = [random.randint(1,100)*np.ones((320), dtype=np.int32) for x in range(perplexity+1)]

    # Example data:
    data_df = pd.DataFrame({
        'img': random_data,
    })

    tsne_result = fit_TSNE(data_df, 'img', n_components, perplexity)
    assert tsne_result.shape == (len(data_df), n_components)

def test_plot_TSNE() -> bool:

    """ Method to test TSNE plot.
    """

    data = np.random.rand(100, 2)
    colors = np.random.rand(100)
    label = ""
    assert plot_TSNE(data, colors, label) == True