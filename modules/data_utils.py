# Imports:
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TSNE:
from sklearn.manifold import TSNE

#################################################################################################################################
def retrieve_data_from_pickle(pickle_file_path:str) -> dict:

    """ Method used to retrieve data emdeddings from pickle file.
        Returns embeddings data dict. 
    """

    # Open the pickle file in read-binary mode:
    with open(pickle_file_path, 'rb') as file:
        # Load the data from the pickle file
        data = pickle.load(file)

    return data   

#################################################################################################################################

def fit_TSNE(data_df:pd.DataFrame, column:str, n_components:int, perplexity:int) -> bool:

    """ Method that receives a pandas dataframe and fits tsne for a selected column data.
    """

    # Instantiate tsne:
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)

    # Let's apply it to our data:
    data_tsne= tsne.fit_transform(np.array(data_df[column].tolist()))

    return data_tsne

#################################################################################################################################

def plot_TSNE(data:list, colors:list, label:str) -> bool:

    """ Plots 2D TSNE for 2 component tsne data and show.
    """

    # Plot the t-SNE of img data:
    fig, axes = plt.subplots(1,1, figsize=(18, 4))

    # Plotting data:
    axes.scatter(data[:, 0], data[:, 1], marker='o', c=colors, edgecolor='k')
    axes.set_title(f't-SNE of Input Data - {label}')
    axes.set_xlabel('t-SNE Component 1')
    axes.set_ylabel('t-SNE Component 2')
    axes.grid(True)

    plt.show()

    return True

#################################################################################################################################



